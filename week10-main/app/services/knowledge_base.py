from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import faiss
import numpy as np
import torch

from app.core.config import settings
from app.models.document import Document, QueryResult
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class KnowledgeBase:
    """Manages Milvus storage and retrieval."""
    index_path: Path | None = None
    metadata_path: Path | None = None
    _rerank_model: AutoModelForSequenceClassification | None = None
    _rerank_tokenizer: AutoTokenizer | None = None
    client: MilvusClient | None = field(init=False, default=None)
    _metadata: list[dict] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.connect()

    def connect(self) -> None:
        if self.client is not None:
            return
        self.client = MilvusClient(settings.mivlus_host)
        if self.client.has_collection(settings.collection_name):
            return
        ## 创建一个支持 混合检索（Hybrid Search） 的向量集合
        # Define Hybrid Search Schema 定义 Schema（表结构）
        # auto_id=False 表示主键 id 需要用户手动提供（而非自增）
        schema = self.client.create_schema(auto_id=False)

        #Primary 字符串型主键，最大长度 64，用于唯一标识文档
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            description="document id"
        )
        # raw text (for BM25 generation and retrieval)
        # 原始文本（用于关键词匹配和 BM25 分析）
        # enable_analyzer=True：Milvus 会自动对 text 进行分词，用于后续生成 BM25 稀疏向量。
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=8192,
            enable_analyzer=True,
            description="raw text content"
        )

        # Dense Vector (CLIP/BERT) 稠密向量（Dense Vector，如 BERT/CLIP 嵌入）
        # dense_vector：文本语义嵌入（如 BERT）
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim = settings.dim,
            description="text embedding"
        )
        # image_vector：图像嵌入（如CLIPvimage encoder）
        schema.add_field(
            field_name="image_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=settings.dim,
            description="image embedding"
        )

        # Sparse Vector (BM25 auto-generated) 稀疏向量（Sparse Vector，由 BM25 自动生成）
        # 类型为 SPARSE_FLOAT_VECTOR，用于存储 BM25 权重（词频、逆文档频率等）,不会手动插入，而是通过函数自动生成。
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
            description="sparse embedding (BM25)"
        )
        # Metadata fields 结构化元数据（如部门标签）
        schema.add_field(
            field_name="department",
            datatype=DataType.VARCHAR,
            max_length=64,
            description="department tag"
        )
        # Dynamic fields for other metadata 允许插入未在 schema 中明确定义的额外字段（类似 JSON 扩展），便于灵活存储元数据。
        schema.enable_dynamic_field = True
        # Add BM25 Function
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        # Create Indices 创建索引（Index）
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",  # or AUTOINDEX
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )
        index_params.add_index(
            field_name="image_vector",
            index_type="HNSW",  # or AUTOINDEX
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",  # Metric for sparse
            params={"drop_ratio_build": 0.2}
        )
        # Create Collection 将 schema 和索引配置一起创建集合。
        # 这是现代 RAG（检索增强生成）系统的典型设计：兼顾语义理解（稠密）与关键词精确匹配（稀疏）
        self.client.create_collection(
            collection_name=settings.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def save(self, data: List[dict]) -> None:
        self.connect()
        formatted_data =[]
        for item in data:
            formatted_item = {
                "id": str(item.get("id")),
                "text": item.get("item") or item.get("answer") or item.get("title") or "",
                "dense_vector": item.get("vector"),
                "department": item.get("department","unknown"),
                # Dynamic fields
                "title": item.get("title", ""),
                "ask": item.get("ask", ""),
                "answer": item.get("answer", ""),
                **{k: v for k, v in item.items() if
                   k not in ["id", "text", "vector", "department", "title", "ask", "answer"]}

            }
            formatted_data.append(formatted_item)

        self.client.insert(
            collection_name=settings.collection_name,
            data=formatted_data,
        )

    def _ensure_rerank_model(self) -> None:
        if self._rerank_model is not None and self._rerank_tokenizer is not None:
            return

        self._rerank_tokenizer = AutoTokenizer.from_pretrained(settings.rerank_model_name)
        self._rerank_model = AutoModelForSequenceClassification.from_pretrained(settings.rerank_model_name)
        self._rerank_model.eval()

    def rerank(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        self._ensure_rerank_model()
        pairs = [[query, candidate] for candidate in candidates]
        with torch.no_grad():
            inputs = self._rerank_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            scores = self._rerank_model(**inputs, return_dict=True).logits.view(-1).float()
            return scores.tolist()


    def extract_tags(self, query: str) -> List[str]:
        """Extract potential tags (e.g., departments) from the query."""
        known_departments = [
            "儿科", "内科", "外科", "妇产科", "骨科", "耳鼻喉科",
            "眼科", "口腔科", "皮肤科", "急诊科", "中医科"
        ]
        tags = []
        for dept in known_departments:
            if dept in query:
                tags.append(dept)
        return tags

    def hybrid_search(
            self,
            query_text: str,
            query_dense_vector: np.ndarray,
            top_k: int = 5,
            rerank: bool = True,
    ) -> list[dict]:
        """
        Perform Hybrid Search (Dense + Sparse/BM25) with RRF Reranking.
        """
        self.connect()

        # 1. Tag Filter
        filter_expr = None
        # 从用户查询中提取“部门”等标签（例如 NER 或关键词匹配）
        tags = self.extract_tags(query_text)
        if tags:
            tags_str = ", ".join([f"'{t}'" for t in tags])
            filter_expr = ''  # f"department in [{tags_str}]"

        #混合检索阶段先召回更多结果（取两者最大值），为后续 rerank 提供足够候选。
        coarse_limit = max(settings.rerank_candidates, top_k)

        # 2. Prepare Search Requests 构建两个 ANN 搜索请求

        # Dense Search Request
        dense_req = AnnSearchRequest(
            data=[query_dense_vector], #查询的嵌入向量（如 BERT 输出）
            anns_field="dense_vector", # 集合中对应的字段
            param={"metric_type": "COSINE", "params": {"nprobe": 10}}, #使用余弦相似度，nprobe=10：HNSW 索引搜索时探索的节点数，影响速度/精度平衡
            limit=coarse_limit,
            expr=filter_expr
        )

        # Sparse Search Request (BM25)
        # Using the server-side BM25 function, we can pass the raw text in the search request
        # if the client and server version support it.
        #在搜索时自动对 query_text 应用 BM25 分析器，生成查询端的稀疏向量，无需客户端预计算
        sparse_req = AnnSearchRequest(
            data=[query_text],    #  # ← 直接传原始文本！
            anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {}},
            limit=coarse_limit,
            expr=filter_expr
        )

        reqs = [dense_req, sparse_req]

        # 3. Execute Hybrid Search
        # Uses RRFRanker (Reciprocal Rank Fusion)  使用 RRFRanker 融合结果
        ranker = RRFRanker(k=60) #k=60 是平滑参数（常用 60）

        # NOTE: MilvusClient.search() validation might fail with AnnSearchRequest list in some versions.
        # We use the underlying Collection object to perform hybrid search.
        from pymilvus import Collection

        # Use the connection established by MilvusClient (usually 'default' or internal alias)
        # MilvusClient(uri=...) sets up a connection.
        # If we encounter issues finding the connection, we might need to explicitly connect,
        # but self.connect() does create a MilvusClient.
        # However, MilvusClient manages its own connection.
        # To use Collection(), we need a registered connection.
        # Since MilvusClient might use a generated alias, let's try to get it or fallback.

        # Safe way: use the client's internal alias if available, or just create a temp Collection with connection reuse
        # But Collection() needs 'using' alias.
        # self.client._using is the alias used by MilvusClient.
        # 某些版本的 milvus-client 对 AnnSearchRequest 列表支持不完善，绕过 MilvusClient 直接使用 Collection
        try:
            col = Collection(settings.collection_name, using=self.client._using) #using=self.client._using 复用 MilvusClient 内部创建的连接别名
            results = col.hybrid_search(
                reqs,
                ranker,
                limit=coarse_limit,
                output_fields=["*"]
            )
        except Exception as e:
            # Fallback if _using is not accessible or other error
            print(f"Hybrid search via Collection failed: {e}. Trying alternative...")
            # If explicit connection is needed (MilvusClient might not register it globally as we expect)
            # This is a fallback but unlikely needed if _using works.
            raise e

        if not results:
            return []

        # hybrid_search 返回 [top_hits]，所以取 results[0]
        hits = results[0]
        if not hits:
            return []

        # 4. Fine Recall (Rerank) 重排序（Rerank）—— 可选精细排序
        if not rerank:
            return [hits[:top_k]]

        #提取候选文本 从每个命中结果中提取原始文本，优先级：text > answer > ask > title
        candidates_text: List[str] = []
        for hit in hits:
            entity = hit.get("entity") or {}
            text = (
                    entity.get("text")
                    or entity.get("answer")
                    or entity.get("ask")
                    or entity.get("title")
                    or ""
            )
            candidates_text.append(text)

        #调用重排序模型
        scores = self.rerank(query_text, candidates_text)

        scored_hits = []
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)
            scored_hits.append(hit)

        scored_hits.sort(key=lambda item: item["rerank_score"], reverse=True)

        return [scored_hits[:top_k]]

    def query(
            self,
            embedding: np.ndarray,
            top_k: int = 5,
            query_text: str | None = None,
    ) -> list[dict]:
        """Wrapper for hybrid search to maintain compatibility."""
        if query_text:
            return self.hybrid_search(query_text, embedding, top_k)
        else:
            # Fallback to simple dense search if no text provided
            self.connect()
            results = self.client.search(
                collection_name=settings.collection_name,
                data=[embedding],
                limit=top_k,
                search_params={"metric_type": "COSINE", "params": {}},
                output_fields=["*"],
            )
            return results

if __name__ == "__main__":
    # 删除数据库
    kg = KnowledgeBase()
    # kg.client.drop_collection(collection_name=settings.collection_name)

##--------------------------------------儿科5-14000.csv 数据入库-------------------------------------
    # # from pathlib import Path
    # # import pandas as pd
    # # data_path = Path(__file__).parent.parent.parent/'data'/'me'/'儿科5-14000.csv'
    # # datas = pd.read_csv(data_path, encoding='gb18030')
    # # print(datas.head())
    # # print(len(datas))
    # # import csv
    # # with open(data_path, 'r',encoding='gb18030') as r:
    # #     reader = csv.reader(r)
    # #
    # #     datas = list()
    # #     try:
    # #         for item in reader:
    # #             datas.append(item)
    # #     except:
    # #         print(f'读取到{len(datas)}条有效数据')
    # # print(" datas[0]", datas[1])
    # # first_text = datas[0]["title"]
    # # print("Type:", type(first_text))
    # # print("Content:", first_text)
    # # print("Is it Chinese?", any("\u4e00" <= c <= "\u9fff" for c in first_text))
    # from app.services.embeddings import EmbeddingService
    # # es = EmbeddingService("sentence-transformers/clip-ViT-B-32")
    # # processed_data = list()
    # # for idx, data in enumerate(datas[1:100]):
    # #     processed_data.append(
    # #         {
    # #             "id": idx,
    # #             'department': data[0],
    # #             "title": data[1],
    # #             "ask": data[2],
    # #             "answer": data[3],
    # #             "vector": es.embed_query(data[2] + data[3])[0]
    # #
    # #         }
    # #     )
    # kg = KnowledgeBase()
    # # kg.save(processed_data)
    # query = '男孩子，已经2岁了，这几天，孩子说自己耳朵又痒又疼，早上，有黄色的耳屎流出，另外，好像没什么食欲也很乏力，请问：孩童中耳炎流黄水要如何治疗。 抗生素药物是目前治疗中耳炎比较常用的，可酌情选。如果孩子情况比较严重的话也可配合一些局部治疗，比如消炎型的滴耳剂，孩子耳痛严重的时候，也是可以适量的使用点止痛的药物，要是伴随发高烧的情况，那么根据孩子的症状使用药物，严重的情况请尽快去医院进行救治，以上都是比较常用的治疗方法，但是如果孩子出现了耳膜穿孔的症状，需要及时的去医院进行手术治疗，治疗期间主要要给孩子做好保暖工作，避免着凉加剧症状。'
    # from app.services.embeddings import EmbeddingService
    #
    # es = EmbeddingService("sentence-transformers/clip-ViT-B-32")
    # # print(es.device)
    # results = kg.query(es.embed_query(query)[0])
    # # print(results)
    # cadidates = []
    # for item in results[0]:
    #     print (item)
    #     print(item['distance'], item['entity']['department'], item['entity']['ask'], item['entity']['answer'])
    #
    # print(cadidates)

####------------------------Corvus-OCR-Caption-Mix 数据入库-----------------------------------
