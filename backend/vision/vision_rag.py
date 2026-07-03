import logging
from backend.vision.visual_retriever import VisualRetriever

logger = logging.getLogger("AgriVisionRAG")


class VisionRAG:
    def __init__(self):
        self.retriever = VisualRetriever()

    def augment_analysis(
        self, smolvlm_result: dict, query_key: str = "disease"
    ) -> dict:
        """
        Takes raw vision outputs and retrieves highly contextual treatment information
        from the indexed agricultural knowledge database.
        """
        # Determine query text
        query_text = smolvlm_result.get(query_key, "")
        if not query_text and "disease" in smolvlm_result:
            query_text = smolvlm_result["disease"]
        elif not query_text:
            query_text = smolvlm_result.get("weed", "")

        # Retrieve documents
        retrieved_docs = []
        if query_text:
            try:
                retrieved_docs = self.retriever.retrieve(query_text, top_k=1)
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")

        # Augment recommendations
        current_recs = smolvlm_result.get("recommendations", [])
        if retrieved_docs:
            rag_info = retrieved_docs[0]["text"]
            # Append retrieved text block as detailed reference recommendation
            if rag_info not in current_recs:
                extended_recs = [f"RAG Advisory: {rag_info}"] + current_recs
                smolvlm_result["recommendations"] = extended_recs

        return smolvlm_result
