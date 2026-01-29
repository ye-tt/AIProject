from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import settings
from app.services.embeddings import get_embedding_service
from app.services.knowledge_base import KnowledgeBase
from app.services.parse_medicin_file import parse_directory
from tqdm import tqdm


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest text files into the knowledge base.")
    parser.add_argument(
        "--source",
        type=Path,
        default=settings.me_data_dir,
        help="Directory containing *.csv files.",
    )
    # parser.add_argument("--chunk-size", type=int, default=300, help="Number of words per chunk.")
    # parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap in words between chunks.")
    return parser

    ## Milvus 中每个字段的最大长度是 65536 字符
def truncate_field(text: str) -> str:
    """截断字段长度"""
    if text is None:
        return ""
    if len(text) <= settings.max_field_length:
        return text
    print(f"警告: 字段长度 {len(text)} 超过限制 {settings.max_field_length}，已截断")
    return text[:settings.max_field_length]

def main() -> None:
    batch_size: int = 500
    args = build_argument_parser().parse_args()
    source_dir: Path = args.source
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")
    print(source_dir)
    all_datas = parse_directory( directory=source_dir)
    embedding_service = get_embedding_service()
    total_items = len(all_datas)
    total_batches = (total_items + batch_size - 1) // batch_size  # 向上取整
    print(f"开始处理 {total_items} 条数据，分 {total_batches} 批，每批 {batch_size} 条")
    knowledge_base = KnowledgeBase()
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_items)
        current_batch = all_datas[start_idx:end_idx]

        processed_batch = []

        # 处理当前批次的数据
        for local_idx, data in enumerate(
                tqdm(current_batch, desc=f"Batch {batch_idx + 1}/{total_batches}", leave=False)):
            global_idx = start_idx + local_idx
            # 截断字段后再生成向量
            truncated_ask = truncate_field(data['ask'])
            truncated_answer = truncate_field(data['answer'])

            # 使用截断后的文本生成向量
            combined_for_embedding = truncated_ask + truncated_answer
            vector = embedding_service.embed_query(combined_for_embedding)[0]

            processed_batch.append(
                {
                    "id": global_idx,
                    'department': data['department'],
                    "title": data['title'],
                    "ask": truncated_ask,
                    "answer":truncated_answer,
                    "vector": vector

                }
            )
        if processed_batch:
            try:
                knowledge_base.save(processed_batch)
            except Exception as e:
                print(processed_batch)
    print(f"Ingested {len(all_datas)} records into the knowledge base.")


if __name__ == "__main__":
    main()
