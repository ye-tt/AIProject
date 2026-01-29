from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from app.services.knowledge_base import KnowledgeBase
from app.services.embeddings import EmbeddingService


@dataclass
class TriageResult:
    department: str
    top_matches: List[Dict]


class TriageService:
    def __init__(self, kb: KnowledgeBase, embedding_service: EmbeddingService):
        self.kb = kb
        self.embedding_service = embedding_service

    def triage(self, symptom_description: str, top_k: int = 1) -> TriageResult:
        """
        基于症状描述进行分诊

        Args:
            symptom_description: 用户描述的症状
            top_k: 返回匹配结果数量

        Returns:
            TriageResult: 分诊结果，包含科室、置信度和匹配记录
        """
        # 阶段1: 粗召回 - 向量检索
        embedding = self.embedding_service.embed_query(symptom_description)[0]
        coarse_results = self.kb.query(
            embedding=embedding,
            query_text=None
        )

        # 阶段2: 精召回 - 重排和融合
        refined_results = self.kb.query(
            embedding=embedding,
            top_k=top_k,
            query_text=symptom_description  # 启用rerank模型
        )

        # 提取科室和分数
        if refined_results and refined_results[0]:
            top_match = refined_results[0][0]
            department = top_match.get("entity", {}).get("department", "未知科室")

            return TriageResult(
                department=department,
                top_matches=self._format_matches(refined_results[0])
            )

        return TriageResult(
            department="请咨询医生",
            top_matches=[]
        )

    def _format_matches(self, matches: List) -> List[Dict]:
        """格式化匹配结果"""
        formatted = []
        for match in matches[:3]:  # 返回top3
            entity = match.get("entity", {})
            formatted.append({
                "department": entity.get("department", ""),
                "title": entity.get("title", ""),
                "distance": match.get("distance", 0.0),
                "rerank_score": match.get("rerank_score", 0.0)
            })
        return formatted


if __name__ == "__main__":
    from app.services.embeddings import EmbeddingService

    kb = KnowledgeBase()

    es = EmbeddingService("sentence-transformers/clip-ViT-B-32")
    triage = TriageService(kb, es)

    # 测试
    test_symptom = "孩子扁桃体炎症发烧该如何治效果好"
    result = triage.triage(test_symptom)

    print(f"建议科室: {result.department}")
    print(f"匹配记录: {result.top_matches}")