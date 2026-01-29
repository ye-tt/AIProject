from app.services.embeddings import EmbeddingService
from app.services.knowledge_base import KnowledgeBase



def triage_with_knowledge_base(query: str, top_k: int = 5):
    embedding_service = EmbeddingService("clip")
    query_vector = embedding_service.embed_query(query)[0]

    kb = KnowledgeBase()
    results = kb.query(query_vector,top_k=top_k,query_text=query)
    if not results or not results[0]:
        return {"department": "未知科室", 'candidates:':[]}

    hits = results[0]
    candidates=[]
    dept_counter = {}
    for hit in hits:
        entity = hit.get("entity",{})
        department = hit.get("department","未知科室")
        dept_counter[department]= dept_counter.get(department,0)+1

        score = float( hit.get("rerank_score", hit.get("distance",0.0)))
        candidates.append({
            "score": score,
            "department": department,
            "title": entity.get("title",""),
            "ask": entity.get("ask",""),
            "answer": entity.get("answer",""),
        })
    best_department = max(dept_counter.items(),key=lambda x: x[1])[0]
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return {
        "department": best_department,
        "candidates": candidates,
    }
if __name__ == "__main__":
    user_query = "男孩子两岁，这几天耳朵又痒又疼，还流黄色分泌物，应挂什么科室？"
    result = triage_with_knowledge_base(user_query, top_k=10)
    print("推荐科室:", result["department"])
    print("Top 候选:")
    for item in result["candidates"][:5]:
        print(item["score"], item["department"], item["title"])
