# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
from typing import List, Dict, Any
from backend.rag.embedding_service import get_embedding

_faiss_index = None
_documents = []

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ml", "models")
INDEX_PATH = os.path.join(MODEL_DIR, "faiss_index.bin")
DOCS_PATH = os.path.join(MODEL_DIR, "faiss_docs.joblib")

def init_kb():
    global _faiss_index, _documents
    if _faiss_index is not None:
        return

    # Seed FAQ documents
    docs = [
        {"text": "Tomato Leaf Mold is caused by Passalora fulva. Symptoms include yellow spots on leaves. Improve greenhouse ventilation.", "disease": "Tomato Leaf Mold"},
        {"text": "Late Blight on Squash is caused by Phytophthora. Dark necrotic lesions appear. Apply copper biological fungicide.", "disease": "Late Blight on Squash"},
        {"text": "Nitrogen deficiency causes uniform yellowing of older leaves. Supplement with organic blood meal or legume compost.", "nutrient": "Nitrogen"},
        {"text": "Potassium deficiency leads to leaf margin curling and necrosis. Apply wood ash or kelp meal.", "nutrient": "Potassium"},
        {"text": "Weeds compete for soil nitrogen and moisture. Use mulching or selective organic pre-emergents.", "weed": "Weed competition"}
    ]

    try:
        import faiss
        # Dimension is 1024 for BGE-M3
        index = faiss.IndexFlatIP(1024)
        embeddings = []
        for doc in docs:
            embeddings.append(get_embedding(doc["text"]))
        
        index.add(np.array(embeddings).astype("float32"))
        _faiss_index = index
        _documents = docs
        
        # Cache index
        os.makedirs(MODEL_DIR, exist_ok=True)
        faiss.write_index(index, INDEX_PATH)
        joblib.dump(docs, DOCS_PATH)
    except Exception:
        # Fallback manual similarity index
        class ManualIndex:
            def __init__(self):
                self.embeddings = []
            def add(self, embs):
                self.embeddings.extend(embs)
            def search(self, query_emb, k):
                # Cosine similarity (dot product of normalized vectors)
                sims = [np.dot(e, query_emb[0]) for e in self.embeddings]
                indices = np.argsort(sims)[::-1][:k]
                return np.array([sims]), np.array([indices])

        index = ManualIndex()
        embeddings = []
        for doc in docs:
            embeddings.append(get_embedding(doc["text"]))
        index.add(embeddings)
        _faiss_index = index
        _documents = docs

def query_kb(query: str, k: int = 2) -> List[Dict[str, Any]]:
    init_kb()
    q_emb = get_embedding(query).reshape(1, -1).astype("float32")
    
    D, I = _faiss_index.search(q_emb, k)
    
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(_documents):
            doc = _documents[idx]
            results.append({
                "text": doc["text"],
                "score": float(score),
                "metadata": doc
            })
    return results
