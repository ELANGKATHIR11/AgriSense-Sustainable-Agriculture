import os
import pickle
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_RAG_LIBS = True
except ImportError:
    HAS_RAG_LIBS = False

from backend.vision.visual_index_builder import DEFAULT_KNOWLEDGE

ENABLE_EMBEDDING_MODEL = os.environ.get("AGRISENSE_ENABLE_RAG_EMBEDDINGS", "false").lower() == "true"

class VisualRetriever:
    def __init__(self, index_dir: str = "ml/models/rag_index"):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "faiss_index.bin")
        self.corpus_path = os.path.join(index_dir, "corpus.pkl")
        self.encoder_name = "all-MiniLM-L6-v2"
        self.model = None
        self.index = None
        self.corpus = DEFAULT_KNOWLEDGE
        
        self.load_index()

    def load_index(self):
        """Loads index and corpus, initializing embedding models."""
        if os.path.exists(self.corpus_path):
            with open(self.corpus_path, "rb") as f:
                self.corpus = pickle.load(f)
        
        if not HAS_RAG_LIBS or not ENABLE_EMBEDDING_MODEL:
            return

        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                self.model = SentenceTransformer(self.encoder_name)
        except Exception as e:
            print(f"Error loading FAISS index: {e}")

    def retrieve(self, query: str, top_k: int = 1) -> list[dict]:
        """Retrieves top_k related manuals or documents for the query."""
        # Fallback if FAISS is not loaded or missing libraries
        if self.index is None or self.model is None or not HAS_RAG_LIBS:
            # Simple keyword matching fallback
            query_lower = query.lower()
            best_match = self.corpus[0]
            max_score = 0
            for doc in self.corpus:
                words = doc["text"].lower().split()
                score = sum(1 for w in words if w in query_lower)
                if score > max_score:
                    max_score = score
                    best_match = doc
            return [best_match]

        # Semantic embedding search
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector, dtype=np.float32), top_k)
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.corpus):
                results.append(self.corpus[idx])
        return results if results else [self.corpus[0]]
