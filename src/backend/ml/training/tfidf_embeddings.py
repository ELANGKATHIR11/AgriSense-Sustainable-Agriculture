#!/usr/bin/env python3
"""
AgriSense TF-IDF Chatbot Embeddings (Python 3.14 Compatible)
============================================================
Simple TF-IDF based embeddings for agricultural chatbot.

Works when sentence-transformers has compatibility issues.

Usage:
    python tfidf_embeddings.py --generate-corpus --output-dir ../data/models/nlp --test

Author: AgriSense ML Team
"""

import os
import sys
import json
import logging
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed. Using sklearn for similarity search.")


class TfidfEmbedder:
    """
    TF-IDF based embedder with optional LSA dimensionality reduction.
    
    Compatible with any Python version.
    """
    
    def __init__(self, 
                 embedding_dim: int = 256,
                 ngram_range: Tuple[int, int] = (1, 2),
                 max_features: int = 10000):
        """
        Args:
            embedding_dim: Output embedding dimension (via SVD)
            ngram_range: N-gram range for TF-IDF
            max_features: Max vocabulary size
        """
        self.embedding_dim = embedding_dim
        self.ngram_range = ngram_range
        self.max_features = max_features
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english',
            sublinear_tf=True  # Use 1 + log(tf) for better results
        )
        
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
        self.is_fitted = False
        
    def fit(self, texts: List[str]):
        """Fit the vectorizer and SVD on texts."""
        logger.info(f"Fitting TF-IDF on {len(texts)} texts...")
        
        # Fit TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Fit SVD for dimensionality reduction
        actual_dim = min(self.embedding_dim, tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        self.svd.fit(tfidf_matrix)
        
        self.is_fitted = True
        logger.info(f"✓ Fitted TF-IDF (vocab={len(self.vectorizer.vocabulary_)}, dim={actual_dim})")
        
    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            normalize: L2 normalize embeddings
            
        Returns:
            Embeddings array (n_texts, embedding_dim)
        """
        if not self.is_fitted:
            raise ValueError("Embedder not fitted. Call fit() first.")
        
        # Transform to TF-IDF
        tfidf_matrix = self.vectorizer.transform(texts)
        
        # Reduce dimensionality
        embeddings = self.svd.transform(tfidf_matrix)
        
        # Normalize
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms
        
        return embeddings.astype('float32')


class SimpleVectorStore:
    """
    Simple vector store using sklearn cosine similarity.
    Falls back when FAISS is not available.
    """
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.documents = []
        self.metadata = []
        self.use_faiss = FAISS_AVAILABLE
        self.index = None
        
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(embedding_dim)
            logger.info("Using FAISS for similarity search")
        else:
            logger.info("Using sklearn for similarity search")
    
    def add(self, 
            embeddings: np.ndarray,
            documents: List[str],
            metadata: List[Dict] = None):
        """Add documents to the store."""
        embeddings = embeddings.astype('float32')
        
        if self.use_faiss:
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        logger.info(f"Added {len(documents)} documents (total: {len(self.documents)})")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        if self.use_faiss:
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx],
                        'score': float(score),
                        'metadata': self.metadata[idx]
                    })
        else:
            # Compute cosine similarity
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                results.append({
                    'text': self.documents[idx],
                    'score': float(similarities[idx]),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def save(self, path: str):
        """Save the vector store."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'embedding_dim': self.embedding_dim,
            'use_faiss': self.use_faiss
        }
        
        if self.use_faiss:
            faiss.write_index(self.index, str(path / 'index.faiss'))
        else:
            data['embeddings'] = self.embeddings
        
        with open(path / 'store.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"✓ Saved vector store to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SimpleVectorStore':
        """Load vector store from disk."""
        path = Path(path)
        
        with open(path / 'store.pkl', 'rb') as f:
            data = pickle.load(f)
        
        store = cls(embedding_dim=data['embedding_dim'])
        store.documents = data['documents']
        store.metadata = data['metadata']
        
        if data['use_faiss'] and FAISS_AVAILABLE:
            store.index = faiss.read_index(str(path / 'index.faiss'))
            store.use_faiss = True
        else:
            store.embeddings = data.get('embeddings')
            store.use_faiss = False
        
        logger.info(f"✓ Loaded vector store from {path}")
        return store


class AgriChatbotKB:
    """
    Agricultural chatbot knowledge base.
    
    Works with Python 3.14 using TF-IDF + SVD.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.embedder = None
        self.vector_store = None
        self.intents = {}
        
    @staticmethod
    def _default_config() -> Dict:
        return {
            'embedding_dim': 256,
            'ngram_range': (1, 2),
            'max_features': 10000,
            'output_dir': './models/nlp'
        }
    
    def load_corpus(self, corpus_path: str) -> Tuple[List[str], List[Dict]]:
        """Load corpus from JSON file."""
        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        metadata = []
        
        # Load intents
        if 'intents' in data:
            self.intents = data['intents']
            for intent_name, intent_data in self.intents.items():
                for pattern in intent_data.get('patterns', []):
                    documents.append(pattern)
                    metadata.append({'intent': intent_name, 'type': 'pattern'})
                for response in intent_data.get('responses', []):
                    documents.append(response)
                    metadata.append({'intent': intent_name, 'type': 'response'})
        
        # Load QA pairs
        if 'qa_pairs' in data:
            for qa in data['qa_pairs']:
                documents.append(qa['question'])
                metadata.append({
                    'intent': qa.get('intent', 'qa'),
                    'answer': qa.get('answer', ''),
                    'type': 'qa'
                })
        
        # Load training data
        if 'training_data' in data:
            for item in data['training_data']:
                documents.append(item['text'])
                metadata.append({
                    'intent': item.get('intent', 'unknown'),
                    'type': 'training'
                })
        
        logger.info(f"Loaded {len(documents)} documents from corpus")
        return documents, metadata
    
    def build_index(self, corpus_path: str):
        """Build the search index."""
        documents, metadata = self.load_corpus(corpus_path)
        
        if not documents:
            logger.warning("No documents to index")
            return
        
        # Initialize embedder
        self.embedder = TfidfEmbedder(
            embedding_dim=self.config['embedding_dim'],
            ngram_range=tuple(self.config['ngram_range']),
            max_features=self.config['max_features']
        )
        
        # Fit and encode
        self.embedder.fit(documents)
        embeddings = self.embedder.encode(documents)
        
        # Build vector store
        actual_dim = embeddings.shape[1]
        self.vector_store = SimpleVectorStore(embedding_dim=actual_dim)
        self.vector_store.add(embeddings, documents, metadata)
        
        logger.info(f"✓ Built index with {len(documents)} documents")
    
    def query(self, query_text: str, k: int = 5) -> Dict:
        """Query the knowledge base."""
        if self.embedder is None or self.vector_store is None:
            raise ValueError("Knowledge base not initialized")
        
        query_embedding = self.embedder.encode([query_text])[0]
        results = self.vector_store.search(query_embedding, k=k)
        
        # Detect intent from results
        intent_scores = {}
        for r in results:
            intent = r['metadata'].get('intent', 'unknown')
            intent_scores[intent] = intent_scores.get(intent, 0) + r['score']
        
        detected_intent = max(intent_scores, key=intent_scores.get) if intent_scores else 'unknown'
        
        return {
            'query': query_text,
            'results': results,
            'detected_intent': detected_intent,
            'confidence': intent_scores.get(detected_intent, 0) / max(1, len(results))
        }
    
    def get_response(self, query_text: str) -> Dict:
        """Get a chatbot response."""
        search_results = self.query(query_text)
        
        response = None
        context = []
        
        for result in search_results['results']:
            meta = result['metadata']
            
            if meta.get('type') == 'qa' and meta.get('answer'):
                response = meta['answer']
                break
            
            if meta.get('type') == 'response':
                response = result['text']
                break
            
            context.append(result['text'])
        
        if not response and context:
            response = f"Related info: {context[0]}"
        elif not response:
            response = "I don't have specific information about that."
        
        return {
            'query': query_text,
            'response': response,
            'intent': search_results['detected_intent'],
            'confidence': search_results['confidence']
        }
    
    def save(self, output_dir: str = None):
        """Save the knowledge base."""
        output_dir = Path(output_dir or self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(output_dir / 'vector_store')
        
        # Save embedder
        with open(output_dir / 'embedder.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': self.embedder.vectorizer,
                'svd': self.embedder.svd,
                'config': self.config
            }, f)
        
        # Save metadata
        with open(output_dir / 'kb_meta.json', 'w', encoding='utf-8') as f:
            json.dump({
                'config': self.config,
                'intents': list(self.intents.keys()),
                'n_documents': len(self.vector_store.documents),
                'created': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"✓ Saved knowledge base to {output_dir}")
    
    @classmethod
    def load(cls, path: str) -> 'AgriChatbotKB':
        """Load knowledge base from disk."""
        path = Path(path)
        
        with open(path / 'kb_meta.json', 'r') as f:
            meta = json.load(f)
        
        kb = cls(config=meta['config'])
        
        # Load embedder
        with open(path / 'embedder.pkl', 'rb') as f:
            emb_data = pickle.load(f)
        
        kb.embedder = TfidfEmbedder(
            embedding_dim=meta['config']['embedding_dim']
        )
        kb.embedder.vectorizer = emb_data['vectorizer']
        kb.embedder.svd = emb_data['svd']
        kb.embedder.is_fitted = True
        
        # Load vector store
        kb.vector_store = SimpleVectorStore.load(path / 'vector_store')
        
        logger.info(f"✓ Loaded knowledge base from {path}")
        return kb


def generate_multilingual_corpus() -> Dict:
    """Generate sample multilingual agricultural corpus."""
    return {
        "qa_pairs": [
            # English
            {"question": "What is the best fertilizer for rice?", 
             "answer": "For rice, use NPK 20:10:10 at 100 kg/ha during transplanting. Apply urea in split doses.",
             "intent": "fertilizer_advice"},
            {"question": "How to control rice blast disease?",
             "answer": "Use Tricyclazole 75% WP at 0.6g/L as foliar spray. Maintain field drainage and avoid excess nitrogen.",
             "intent": "disease_detection"},
            {"question": "Best time to sow wheat in Punjab?",
             "answer": "Optimal sowing time for wheat in Punjab is November 10-25 for timely sown varieties.",
             "intent": "crop_recommendation"},
            {"question": "How much water does rice need?",
             "answer": "Rice requires 1200-1500mm water during the growing season. Maintain 5cm standing water during tillering.",
             "intent": "irrigation_advice"},
            {"question": "What causes yellowing of rice leaves?",
             "answer": "Yellowing can be caused by nitrogen deficiency, iron chlorosis, or bacterial leaf blight. Check nutrient levels first.",
             "intent": "disease_detection"},
            
            # Hindi (Romanized for TF-IDF compatibility)
            {"question": "Dhaan ke liye sabse achha urvarak kaunsa hai?",
             "answer": "Dhaan ke liye NPK 20:10:10 ka prayog 100 kg/hectare ki dar se ropai ke samay karein.",
             "intent": "fertilizer_advice"},
            {"question": "Gehu ki buwai ka sabse achha samay kya hai?",
             "answer": "Punjab mein gehu bone ka sabse achha samay 10-25 November hai.",
             "intent": "crop_recommendation"},
            
            # Tamil (Romanized)
            {"question": "Nellukku sirandha uram ethu?",
             "answer": "Nellukku NPK 20:10:10 ai 100 kilo/hectare vidhaththil nadavu seyyumpoḻuthu payanpadutthavum.",
             "intent": "fertilizer_advice"},
            
            # More agriculture topics
            {"question": "How to identify late blight in potato?",
             "answer": "Late blight shows water-soaked lesions on leaves that turn brown. White fungal growth appears under leaves in humid conditions.",
             "intent": "disease_detection"},
            {"question": "Best fertilizer for tomato?",
             "answer": "Apply NPK 60:80:60 kg/ha as basal dose. Top dress with 30kg nitrogen at flowering and fruiting stages.",
             "intent": "fertilizer_advice"},
            {"question": "How to control aphids in mustard?",
             "answer": "Spray Dimethoate 30EC at 1ml/L or use neem oil 5ml/L. Encourage natural predators like ladybugs.",
             "intent": "pest_control"},
            {"question": "When to harvest wheat?",
             "answer": "Harvest wheat when grains are hard, moisture content is 14-15%, and stalks turn golden yellow.",
             "intent": "harvest_advice"},
            {"question": "Soil pH for cotton cultivation?",
             "answer": "Cotton grows best in soil pH 6.0-7.5. Apply lime if pH is below 6.0 or gypsum if above 8.0.",
             "intent": "soil_advice"},
        ],
        "intents": {
            "crop_recommendation": {
                "patterns": [
                    "What crop should I grow",
                    "Which crop is best for my soil",
                    "Suggest crops for my region",
                    "Best crop for summer",
                    "What to plant in monsoon",
                    "Kaun si fasal ugaaun"
                ],
                "responses": [
                    "Based on your soil and climate conditions, I recommend considering the regional crop calendar.",
                    "The best crop depends on soil type, water availability, and market conditions."
                ]
            },
            "fertilizer_advice": {
                "patterns": [
                    "How much fertilizer to use",
                    "NPK ratio for crops",
                    "When to apply urea",
                    "Fertilizer schedule",
                    "Urvarak kitna dena chahiye"
                ],
                "responses": [
                    "Fertilizer recommendations depend on soil test results and crop requirements.",
                    "Always conduct a soil test before applying fertilizers."
                ]
            },
            "disease_detection": {
                "patterns": [
                    "My plant has disease",
                    "Leaves are turning yellow",
                    "How to treat leaf spots",
                    "Brown spots on leaves",
                    "Plant wilting problem"
                ],
                "responses": [
                    "Please describe the symptoms in detail or share a photo for accurate diagnosis.",
                    "Common diseases can be prevented with proper spacing and fungicide application."
                ]
            },
            "pest_control": {
                "patterns": [
                    "How to control pests",
                    "Insects eating my crops",
                    "Caterpillar damage",
                    "Aphid infestation",
                    "Pest management"
                ],
                "responses": [
                    "Integrated pest management combining cultural, biological and chemical methods works best.",
                    "Identify the pest first before applying any pesticide."
                ]
            },
            "irrigation_advice": {
                "patterns": [
                    "How much water needed",
                    "Irrigation schedule",
                    "When to water crops",
                    "Drip irrigation setup"
                ],
                "responses": [
                    "Water requirements vary by crop stage and weather conditions.",
                    "Morning irrigation is generally more effective."
                ]
            }
        },
        "training_data": [],
        "metadata": {
            "languages": ["en", "hi", "ta"],
            "created": datetime.now().isoformat()
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Build TF-IDF embeddings for AgriSense chatbot')
    parser.add_argument('--corpus', type=str, default=None, help='Path to corpus JSON')
    parser.add_argument('--output-dir', type=str, default='./models/nlp')
    parser.add_argument('--generate-corpus', action='store_true', help='Generate sample corpus')
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--test', action='store_true', help='Run test queries')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🌐 AgriSense TF-IDF Chatbot Embeddings")
    print("   (Python 3.14 Compatible)")
    print("=" * 70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate corpus if needed
    if args.generate_corpus or not args.corpus:
        logger.info("Generating multilingual corpus...")
        corpus = generate_multilingual_corpus()
        
        # Add training data from intent patterns
        for intent_name, intent_data in corpus['intents'].items():
            for pattern in intent_data['patterns']:
                corpus['training_data'].append({
                    'text': pattern,
                    'intent': intent_name
                })
        
        corpus_path = output_dir / 'agri_corpus.json'
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Generated corpus: {corpus_path}")
        args.corpus = str(corpus_path)
    
    # Build knowledge base
    kb = AgriChatbotKB({
        'embedding_dim': args.embedding_dim,
        'ngram_range': [1, 2],
        'max_features': 10000,
        'output_dir': args.output_dir
    })
    
    logger.info(f"Building index from: {args.corpus}")
    kb.build_index(args.corpus)
    kb.save(args.output_dir)
    
    # Test queries
    if args.test:
        print("\n" + "=" * 70)
        print("🧪 Testing Queries")
        print("=" * 70)
        
        test_queries = [
            "What fertilizer should I use for rice?",
            "How to control plant disease?",
            "Best time to plant wheat",
            "Dhaan ke liye kaunsa urvarak",
            "My leaves are turning yellow",
            "How much water for cotton?"
        ]
        
        for query in test_queries:
            result = kb.get_response(query)
            print(f"\n📝 Query: {query}")
            print(f"   Intent: {result['intent']} (conf: {result['confidence']:.2f})")
            print(f"   Response: {result['response'][:80]}...")
    
    print("\n" + "=" * 70)
    print("✅ TF-IDF Embeddings Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    print(f"  - agri_corpus.json")
    print(f"  - vector_store/")
    print(f"  - embedder.pkl")
    print(f"  - kb_meta.json")


if __name__ == '__main__':
    main()
