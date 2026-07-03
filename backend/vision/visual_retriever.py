# -*- coding: utf-8 -*-
from backend.rag.mrag_orchestrator import mrag_orchestrator


class VisualRetriever:
    def __init__(self, index_dir=None):
        pass

    def retrieve(self, query: str, top_k: int = 1) -> list[dict]:
        results = mrag_orchestrator.search_collection(
            collection_name="documents", query=query, k=top_k
        )
        return [{"text": r["text"], "metadata": r.get("metadata")} for r in results]
