import os
import pickle
import numpy as np

# Try to import faiss and sentence_transformers
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_RAG_LIBS = True
except ImportError:
    HAS_RAG_LIBS = False

# Default knowledge corpus of treatment protocols and organic manuals
DEFAULT_KNOWLEDGE = [
    {"title": "Tomato Leaf Mold Treatment", "text": "Tomato Leaf Mold is caused by Passalora fulva. Treat by reducing greenhouse humidity below 85%, increasing spacing for ventilation, and applying copper fungicides or bio-fungicides containing Bacillus subtilis."},
    {"title": "Powdery Mildew Squash Protocol", "text": "Powdery mildew thrives in dry foliage under high humidity. Apply organic horticultural oils, neem oil, or potassium bicarbonate. Ensure squash crops are planted in full sun with adequate air circulation."},
    {"title": "Pigweed Weed Management", "text": "Amaranthus retroflexus (Pigweed) is an aggressive competitor. Suppress using deep organic wood mulch or straw covers. Extract manually before seed dispersal or use pre-emergents like corn gluten meal."},
    {"title": "Nitrogen Deficiency Correction", "text": "Nitrogen deficiency causes yellowing of older leaves (chlorosis). Correct by applying liquid fish emulsion, blood meal, compost tea, or planting cover crops like legumes (clover, vetch) to fix atmospheric nitrogen."},
    {"title": "Phosphorus Deficiency correction", "text": "Phosphorus deficiency is marked by purple or dark green leaf tints and stunted growth. Apply bone meal, rock phosphate, or organic compost. Ensure soil pH is between 6.0 and 7.0 for optimal absorption."},
    {"title": "Potassium Deficiency remedy", "text": "Potassium deficiency causes leaf edge scorching and necrosis. Apply wood ash, greensand, or organic kelp meal. Potassium aids in cellular turgor and defense against crop pathogens."}
]

class VisualIndexBuilder:
    def __init__(self, index_dir: str = "ml/models/rag_index"):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "faiss_index.bin")
        self.corpus_path = os.path.join(index_dir, "corpus.pkl")
        self.encoder_name = "all-MiniLM-L6-v2"

    def build_and_save_index(self):
        """Encodes the knowledge text corpus and creates a FAISS flat index."""
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Save corpus
        with open(self.corpus_path, "wb") as f:
            pickle.dump(DEFAULT_KNOWLEDGE, f)
        
        if not HAS_RAG_LIBS:
            print("FAISS or SentenceTransformers not installed. Skipping binary index serialization.")
            return

        print(f"Loading embedding model: {self.encoder_name}...")
        model = SentenceTransformer(self.encoder_name)
        
        texts = [doc["text"] for doc in DEFAULT_KNOWLEDGE]
        embeddings = model.encode(texts)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        
        faiss.write_index(index, self.index_path)
        print(f"FAISS index successfully saved to {self.index_path}")

if __name__ == "__main__":
    builder = VisualIndexBuilder()
    builder.build_and_save_index()
