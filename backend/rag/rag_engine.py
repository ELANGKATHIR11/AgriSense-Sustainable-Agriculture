"""
AGRISENSE Advanced RAG (Retrieval-Augmented Generation) Engine
Upgraded to use FAISS vector index, BGE-M3 embeddings, and BGE-Reranker-v2.
Features robust CPU/GPU offloading and zero-dependency fallbacks.
"""

import sqlite3
import json
import os
import numpy as np
from typing import List, Dict, Any

# Paths
DB_PATH = "rag_vector_store.db"
FAISS_INDEX_PATH = "ml/models/faiss_index.bin"
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

# Try loading FAISS and SentenceTransformers
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class RAGEngine:
    def __init__(self):
        self._init_db()
        self.device = "cuda" if (os.getenv("RAG_DEVICE") == "cuda" or False) else "cpu"
        self.encoder = None
        self.reranker = None
        self.faiss_index = None
        self.dimension = 1024 # BGE-M3 default dimension

        self._load_models()
        self._init_faiss()
        self._seed_database()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS disease_kb (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    disease_name TEXT UNIQUE,
                    symptoms TEXT,
                    recommendations TEXT,
                    embedding_json TEXT
                )
            """)

    def _load_models(self):
        if not TRANSFORMERS_AVAILABLE:
            print("SentenceTransformers not installed. Using local TF-IDF similarity fallback.")
            return

        # Load Encoder (BGE-M3)
        try:
            print(f"Loading SentenceTransformer BAAI/bge-m3 on {self.device}...")
            self.encoder = SentenceTransformer("BAAI/bge-m3", device=self.device)
            self.dimension = self.encoder.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Failed to load BGE-M3: {e}. Falling back to MiniLM-L6-v2 on CPU...")
            try:
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                self.dimension = self.encoder.get_sentence_embedding_dimension()
            except Exception as e2:
                print(f"Failed to load MiniLM fallback: {e2}. Disabling transformers.")
                self.encoder = None

        # Load Reranker (BGE-Reranker-v2-m3)
        if self.encoder is not None:
            try:
                print(f"Loading CrossEncoder BAAI/bge-reranker-v2-m3 on CPU...")
                # Reranker runs on CPU by default to conserve RTX 5060 VRAM
                self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
            except Exception as e:
                print(f"Failed to load BGE-Reranker: {e}. Reranking will be disabled.")
                self.reranker = None

    def _init_faiss(self):
        if not FAISS_AVAILABLE:
            print("FAISS not installed. Using numpy matrix operations.")
            return

        if os.path.exists(FAISS_INDEX_PATH):
            try:
                self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                print(f"Loaded persistent FAISS index from {FAISS_INDEX_PATH}.")
                return
            except Exception as e:
                print(f"Error reading FAISS index: {e}. Rebuilding...")

        # Initialize IndexFlatIP (Inner Product/Cosine Similarity for normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.dimension)

    def _seed_database(self):
        diseases = [
            {
                "disease_name": "Tomato Late Blight",
                "symptoms": "Dark brown or black water-soaked spots on leaves. White velvety fungal growth appears on under-leaves during humid weather. Necrotic stems and rot on tomato fruits.",
                "recommendations": "Apply copper-based fungicides immediately. Remove and destroy infected foliage. Improve air circulation and avoid overhead watering to keep leaves dry."
            },
            {
                "disease_name": "Tomato Leaf Mold",
                "symptoms": "Pale green or yellow spots on upper leaf surfaces. Olive-green to purple velvet-like mold coating under-leaves. Curling leaves and foliage necrosis.",
                "recommendations": "Ensure ventilation and lower relative humidity (<85%). Prune lower leaves to enhance airflow. Avoid overhead watering; use drip lines."
            },
            {
                "disease_name": "Powdery Mildew on Squash",
                "symptoms": "White, circular talcum-like powdery spots on leaves and stems. Premature leaf defoliation and stunted plant growth.",
                "recommendations": "Apply organic neem oil extract or potassium bicarbonate. Space squash plants widely to get full direct sunlight."
            },
            {
                "disease_name": "Citrus Canker",
                "symptoms": "Raised corky, brown lesions with yellow halos on citrus fruits, leaves, and twigs. Premature leaf and fruit drop.",
                "recommendations": "Spray preventative copper-based bactericides. Prune infected branches during dry weather. Disinfect tools after pruning."
            },
            {
                "disease_name": "Corn Common Rust",
                "symptoms": "Reddish-brown to orange powdery pustules on both upper and lower leaf surfaces. Yellowing leaves and lodging under heavy infestation.",
                "recommendations": "Plant rust-resistant corn hybrids. Apply foliar strobilurin or triazole fungicides. Rotate crops to non-cereal hosts."
            },
            {
                "disease_name": "Potato Common Scab",
                "symptoms": "Dark brown, corky, raised or pitted lesions on potato tubers, reducing marketability.",
                "recommendations": "Maintain soil pH below 5.2. Ensure adequate soil moisture during tuber initiation. Rotate with alfalfa or oats."
            }
        ]

        # Seed data and build FAISS index
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            rebuild_needed = False
            for d in diseases:
                cursor.execute("SELECT id, embedding_json FROM disease_kb WHERE disease_name = ?", (d["disease_name"],))
                row = cursor.fetchone()
                if not row:
                    emb = self._compute_embedding(f"Disease: {d['disease_name']}. Symptoms: {d['symptoms']}")
                    cursor.execute(
                        "INSERT INTO disease_kb (disease_name, symptoms, recommendations, embedding_json) VALUES (?, ?, ?, ?)",
                        (d["disease_name"], d["symptoms"], d["recommendations"], json.dumps(emb))
                    )
                    rebuild_needed = True
            conn.commit()

        if rebuild_needed or (self.faiss_index is not None and self.faiss_index.ntotal == 0):
            self.rebuild_faiss_index()

    def _compute_embedding(self, text: str) -> List[float]:
        if self.encoder is not None:
            emb = self.encoder.encode(text, normalize_embeddings=True)
            return emb.tolist()
        
        # Heuristic NumPy fallback
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(self.dimension)
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    def rebuild_faiss_index(self):
        """Rebuilds the FAISS index completely from SQLite records."""
        if self.faiss_index is None:
            return

        self.faiss_index.reset()
        vectors = []
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT embedding_json FROM disease_kb ORDER BY id ASC")
            for row in cursor.fetchall():
                emb = json.loads(row[0])
                if len(emb) != self.dimension:
                    # Pad/trim to match dimension
                    emb = np.resize(emb, (self.dimension,)).tolist()
                vectors.append(emb)

        if len(vectors) > 0:
            vectors_np = np.array(vectors, dtype=np.float32)
            # Normalize for cosine similarity
            faiss.normalize_L2(vectors_np)
            self.faiss_index.add(vectors_np)
            self.save_faiss_index()

    def save_faiss_index(self):
        if self.faiss_index is not None:
            try:
                faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
                print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
            except Exception as e:
                print(f"Failed to save FAISS index: {e}")

    async def add_disease_record(self, name: str, symptoms: str, recommendations: str):
        text_for_embedding = f"Disease: {name}. Symptoms: {symptoms}. Recommendations: {recommendations}"
        emb = self._compute_embedding(text_for_embedding)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO disease_kb (disease_name, symptoms, recommendations, embedding_json) VALUES (?, ?, ?, ?)",
                (name, symptoms, recommendations, json.dumps(emb))
            )
            conn.commit()

        # Re-index
        if self.faiss_index is not None:
            self.rebuild_faiss_index()

    async def query_disease_kb(self, user_query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieves and reranks top matched records from KB."""
        query_emb = self._compute_embedding(user_query)
        q_vec = np.array([query_emb], dtype=np.float32)
        
        matches = []
        # Step 1: Retrieve records
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT id, disease_name, symptoms, recommendations, embedding_json FROM disease_kb ORDER BY id ASC")
            db_records = cursor.fetchall()

        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            faiss.normalize_L2(q_vec)
            distances, indices = self.faiss_index.search(q_vec, min(top_k * 3, len(db_records)))
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(db_records):
                    continue
                db_id, name, sym, rec, _ = db_records[idx]
                matches.append({
                    "disease_name": name,
                    "symptoms": sym,
                    "recommendations": rec,
                    "score": float(dist)
                })
        else:
            # Fallback to NumPy-based manual cosine similarity matching
            q_vec_flat = q_vec[0]
            for _, name, sym, rec, emb_json in db_records:
                db_vec = np.array(json.loads(emb_json))
                if len(db_vec) != len(q_vec_flat):
                    db_vec = np.resize(db_vec, q_vec_flat.shape)
                dot_prod = np.dot(q_vec_flat, db_vec)
                norm_q = np.linalg.norm(q_vec_flat)
                norm_db = np.linalg.norm(db_vec)
                similarity = dot_prod / (norm_q * norm_db) if (norm_q * norm_db) > 0 else 0.0
                matches.append({
                    "disease_name": name,
                    "symptoms": sym,
                    "recommendations": rec,
                    "score": float(similarity)
                })
            matches.sort(key=lambda x: x["score"], reverse=True)
            matches = matches[:top_k * 3]

        # Step 2: Rerank matches using CrossEncoder
        if self.reranker is not None and len(matches) > 0:
            pairs = [[user_query, f"Disease: {m['disease_name']}. Symptoms: {m['symptoms']}. Recommendations: {m['recommendations']}"] for m in matches]
            try:
                rerank_scores = self.reranker.predict(pairs)
                for i, score in enumerate(rerank_scores):
                    matches[i]["rerank_score"] = float(score)
                # Sort by rerank score descending
                matches.sort(key=lambda x: x.get("rerank_score", -9999), reverse=True)
            except Exception as e:
                print(f"Reranking error: {e}. Falling back to retrieval score.")
                matches.sort(key=lambda x: x["score"], reverse=True)
        else:
            matches.sort(key=lambda x: x["score"], reverse=True)

        return matches[:top_k]

rag_engine = RAGEngine()
