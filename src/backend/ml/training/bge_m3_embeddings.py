#!/usr/bin/env python3
"""
AgriSense BGE-M3 Multilingual Embeddings
=========================================
Generates multilingual embeddings for agricultural chatbot using BGE-M3.

Features:
- BAAI/bge-m3 for Hindi/Tamil/English
- FAISS vector store for similarity search
- Dense + Sparse hybrid retrieval
- Knowledge base indexing
- RAG-ready output

Usage:
    python bge_m3_embeddings.py --corpus ../data/intent_corpus/intents.json

Author: AgriSense ML Team
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import pickle

import numpy as np

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed")

SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("sentence-transformers loaded successfully")
except ImportError as e:
    logger.warning(f"sentence-transformers not installed: {e}")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed. Install with: pip install faiss-cpu")

try:
    from FlagEmbedding import BGEM3FlagModel
    BGEM3_AVAILABLE = True
except ImportError:
    BGEM3_AVAILABLE = False
    logger.info("FlagEmbedding not installed, using sentence-transformers fallback")


class BGEM3Embedder:
    """
    BGE-M3 multilingual embedder with hybrid retrieval support.
    
    Supports:
    - Dense embeddings (1024-dim)
    - Sparse embeddings (lexical)
    - ColBERT multi-vector
    """
    
    def __init__(self, 
                 model_name: str = 'BAAI/bge-m3',
                 use_fp16: bool = True,
                 device: str = None):
        """
        Args:
            model_name: HuggingFace model name
            use_fp16: Use half precision
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.embedding_dim = 1024  # BGE-M3 dimension
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        logger.info(f"Loading BGE-M3 model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        if BGEM3_AVAILABLE:
            # Use FlagEmbedding for full BGE-M3 features
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device
            )
            self.model_type = 'flag'
            logger.info("✓ Loaded FlagEmbedding BGE-M3 (full features)")
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            # Fallback to sentence-transformers
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model_type = 'sentence_transformers'
            logger.info("✓ Loaded sentence-transformers BGE-M3")
        else:
            raise ImportError("Either FlagEmbedding or sentence-transformers required")
    
    def encode(self, 
               texts: List[str],
               batch_size: int = 32,
               return_sparse: bool = False,
               return_colbert: bool = False,
               normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            return_sparse: Return sparse embeddings (lexical)
            return_colbert: Return ColBERT multi-vector
            normalize: L2 normalize embeddings
            
        Returns:
            Dict with 'dense' embeddings and optionally 'sparse', 'colbert'
        """
        if not texts:
            return {'dense': np.array([])}
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        if self.model_type == 'flag' and BGEM3_AVAILABLE:
            # Full BGE-M3 encoding
            outputs = self.model.encode(
                texts,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert
            )
            
            result = {'dense': outputs['dense_vecs']}
            
            if return_sparse and 'lexical_weights' in outputs:
                result['sparse'] = outputs['lexical_weights']
            if return_colbert and 'colbert_vecs' in outputs:
                result['colbert'] = outputs['colbert_vecs']
        else:
            # Sentence-transformers encoding
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=True
            )
            result = {'dense': embeddings}
        
        logger.info(f"✓ Encoded {len(texts)} texts to {result['dense'].shape}")
        return result
    
    def encode_query(self, query: str) -> Dict[str, np.ndarray]:
        """Encode a single query (optimized for retrieval)."""
        return self.encode([query], return_sparse=True)


class FAISSVectorStore:
    """
    FAISS-based vector store for similarity search.
    
    Supports:
    - Flat (exact) search
    - IVF (approximate) search
    - HNSW (graph-based) search
    """
    
    def __init__(self, 
                 embedding_dim: int = 1024,
                 index_type: str = 'flat',
                 nlist: int = 100):
        """
        Args:
            embedding_dim: Dimension of embeddings
            index_type: 'flat', 'ivf', or 'hnsw'
            nlist: Number of clusters for IVF
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss required. Install with: pip install faiss-cpu")
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index."""
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine sim)
        elif self.index_type == 'ivf':
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
        elif self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # M=32
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"Created FAISS {self.index_type} index (dim={self.embedding_dim})")
    
    def add(self, 
            embeddings: np.ndarray,
            documents: List[str],
            metadata: List[Dict] = None):
        """
        Add documents to the index.
        
        Args:
            embeddings: Document embeddings
            documents: Document texts
            metadata: Optional metadata for each document
        """
        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents must have same length")
        
        # Normalize for cosine similarity
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Train IVF index if needed
        if self.index_type == 'ivf' and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        logger.info(f"Added {len(documents)} documents (total: {len(self.documents)})")
    
    def search(self,
               query_embedding: np.ndarray,
               k: int = 5) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            k: Number of results
            
        Returns:
            List of dicts with 'text', 'score', 'metadata'
        """
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'score': float(score),
                    'metadata': self.metadata[idx],
                    'index': int(idx)
                })
        
        return results
    
    def save(self, path: str):
        """Save index and documents."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / 'index.faiss'))
        
        # Save documents and metadata
        with open(path / 'documents.pkl', 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"✓ Saved vector store to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FAISSVectorStore':
        """Load index from disk."""
        path = Path(path)
        
        with open(path / 'documents.pkl', 'rb') as f:
            data = pickle.load(f)
        
        store = cls(
            embedding_dim=data['embedding_dim'],
            index_type=data['index_type']
        )
        store.index = faiss.read_index(str(path / 'index.faiss'))
        store.documents = data['documents']
        store.metadata = data['metadata']
        
        logger.info(f"✓ Loaded vector store from {path} ({len(store.documents)} docs)")
        return store


class AgriKnowledgeBase:
    """
    Agricultural knowledge base with RAG support.
    
    Features:
    - Multilingual query support (Hindi/Tamil/English)
    - Intent classification
    - Semantic search
    - Contextual retrieval
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.embedder = None
        self.vector_store = None
        self.intents = {}
        self.qa_pairs = []
        
    @staticmethod
    def _default_config() -> Dict:
        return {
            'model_name': 'BAAI/bge-m3',
            'embedding_dim': 1024,
            'index_type': 'flat',
            'batch_size': 32,
            'output_dir': './models/nlp',
            'languages': ['en', 'hi', 'ta']  # English, Hindi, Tamil
        }
    
    def initialize(self):
        """Initialize embedder and vector store."""
        self.embedder = BGEM3Embedder(
            model_name=self.config['model_name']
        )
        
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.config['embedding_dim'],
            index_type=self.config['index_type']
        )
    
    def load_corpus(self, corpus_path: str) -> int:
        """
        Load knowledge corpus from JSON file.
        
        Expected format:
        {
            "intents": {...},
            "training_data": [...],
            "qa_pairs": [...]  # Optional
        }
        """
        corpus_path = Path(corpus_path)
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load intents
        if 'intents' in data:
            self.intents = data['intents']
            logger.info(f"Loaded {len(self.intents)} intent categories")
        
        # Load training data
        documents = []
        metadata = []
        
        if 'training_data' in data:
            for item in data['training_data']:
                documents.append(item['text'])
                metadata.append({
                    'intent': item.get('intent', 'unknown'),
                    'type': 'training'
                })
        
        # Load QA pairs
        if 'qa_pairs' in data:
            for qa in data['qa_pairs']:
                documents.append(qa['question'])
                metadata.append({
                    'answer': qa.get('answer', ''),
                    'intent': qa.get('intent', 'qa'),
                    'type': 'qa'
                })
                self.qa_pairs.append(qa)
        
        # Add intent patterns and responses
        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data.get('patterns', []):
                documents.append(pattern)
                metadata.append({
                    'intent': intent_name,
                    'type': 'pattern'
                })
            
            for response in intent_data.get('responses', []):
                documents.append(response)
                metadata.append({
                    'intent': intent_name,
                    'type': 'response'
                })
        
        logger.info(f"Total documents: {len(documents)}")
        return len(documents), documents, metadata
    
    def build_index(self, corpus_path: str):
        """Build vector index from corpus."""
        if self.embedder is None:
            self.initialize()
        
        # Load corpus
        n_docs, documents, metadata = self.load_corpus(corpus_path)
        
        if n_docs == 0:
            logger.warning("No documents to index")
            return
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.encode(
            documents,
            batch_size=self.config['batch_size']
        )['dense']
        
        # Add to vector store
        self.vector_store.add(embeddings, documents, metadata)
        
        logger.info(f"✓ Built index with {n_docs} documents")
    
    def query(self, 
              query_text: str,
              k: int = 5,
              intent_filter: str = None) -> Dict:
        """
        Query the knowledge base.
        
        Args:
            query_text: Query string (any language)
            k: Number of results
            intent_filter: Optional intent filter
            
        Returns:
            Dict with results and detected intent
        """
        if self.embedder is None or self.vector_store is None:
            raise ValueError("Knowledge base not initialized")
        
        # Encode query
        query_embedding = self.embedder.encode([query_text])['dense'][0]
        
        # Search
        results = self.vector_store.search(query_embedding, k=k * 2)  # Get more for filtering
        
        # Filter by intent if specified
        if intent_filter:
            results = [r for r in results if r['metadata'].get('intent') == intent_filter]
        
        # Detect intent from top results
        intent_votes = {}
        for r in results[:k]:
            intent = r['metadata'].get('intent', 'unknown')
            intent_votes[intent] = intent_votes.get(intent, 0) + r['score']
        
        detected_intent = max(intent_votes, key=intent_votes.get) if intent_votes else 'unknown'
        
        return {
            'query': query_text,
            'results': results[:k],
            'detected_intent': detected_intent,
            'intent_confidence': intent_votes.get(detected_intent, 0) / k if k > 0 else 0
        }
    
    def get_response(self, query_text: str) -> Dict:
        """
        Get a response for a query (RAG-style).
        
        Args:
            query_text: User query
            
        Returns:
            Dict with response and context
        """
        search_results = self.query(query_text)
        
        # Find best response
        response = None
        context = []
        
        for result in search_results['results']:
            meta = result['metadata']
            
            # If it's a QA pair, return the answer
            if meta.get('type') == 'qa' and meta.get('answer'):
                response = meta['answer']
                break
            
            # If it's a response template
            if meta.get('type') == 'response':
                response = result['text']
                break
            
            # Collect context
            context.append(result['text'])
        
        # Default response if none found
        if not response and context:
            response = f"Based on: {context[0]}"
        elif not response:
            response = "I don't have specific information about that. Please ask about crops, diseases, or farming practices."
        
        return {
            'query': query_text,
            'response': response,
            'intent': search_results['detected_intent'],
            'confidence': search_results['intent_confidence'],
            'context': context[:3]
        }
    
    def save(self, output_dir: str = None):
        """Save knowledge base."""
        output_dir = Path(output_dir or self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(output_dir / 'vector_store')
        
        # Save intents and config
        with open(output_dir / 'knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump({
                'config': self.config,
                'intents': self.intents,
                'qa_pairs': self.qa_pairs,
                'num_documents': len(self.vector_store.documents)
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved knowledge base to {output_dir}")
    
    @classmethod
    def load(cls, path: str) -> 'AgriKnowledgeBase':
        """Load knowledge base from disk."""
        path = Path(path)
        
        with open(path / 'knowledge_base.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kb = cls(config=data['config'])
        kb.intents = data['intents']
        kb.qa_pairs = data['qa_pairs']
        
        # Initialize embedder
        kb.embedder = BGEM3Embedder(model_name=kb.config['model_name'])
        
        # Load vector store
        kb.vector_store = FAISSVectorStore.load(path / 'vector_store')
        
        logger.info(f"✓ Loaded knowledge base from {path}")
        return kb


def generate_multilingual_corpus() -> Dict:
    """Generate sample multilingual agricultural corpus."""
    corpus = {
        "qa_pairs": [
            # English
            {"question": "What is the best fertilizer for rice?", 
             "answer": "For rice, use NPK 20:10:10 at 100 kg/ha during transplanting.",
             "intent": "fertilizer_advice"},
            {"question": "How to control rice blast disease?",
             "answer": "Use Tricyclazole 75% WP at 0.6g/L as foliar spray. Maintain field drainage.",
             "intent": "disease_detection"},
            {"question": "Best time to sow wheat in Punjab?",
             "answer": "Optimal sowing time for wheat in Punjab is November 10-25.",
             "intent": "crop_recommendation"},
            
            # Hindi
            {"question": "धान के लिए सबसे अच्छा उर्वरक कौन सा है?",
             "answer": "धान के लिए NPK 20:10:10 का प्रयोग 100 किग्रा/हेक्टेयर की दर से रोपाई के समय करें।",
             "intent": "fertilizer_advice"},
            {"question": "धान का झुलसा रोग कैसे नियंत्रित करें?",
             "answer": "ट्राइसाइक्लाजोल 75% WP का 0.6 ग्राम/लीटर पानी में घोलकर छिड़काव करें।",
             "intent": "disease_detection"},
            {"question": "गेहूं की बुवाई का सबसे अच्छा समय क्या है?",
             "answer": "पंजाब में गेहूं बोने का सबसे अच्छा समय 10-25 नवंबर है।",
             "intent": "crop_recommendation"},
            
            # Tamil
            {"question": "நெல்லுக்கு சிறந்த உரம் எது?",
             "answer": "நெல்லுக்கு NPK 20:10:10 ஐ 100 கிலோ/ஹெக்டேர் விகிதத்தில் நடவு செய்யும்போது பயன்படுத்தவும்.",
             "intent": "fertilizer_advice"},
            {"question": "நெல் கருகல் நோயை எப்படி கட்டுப்படுத்துவது?",
             "answer": "ட்ரைசைக்ளாசோல் 75% WP ஐ 0.6 கிராம்/லிட்டர் என்ற விகிதத்தில் தெளிக்கவும்.",
             "intent": "disease_detection"},
        ],
        "intents": {
            "crop_recommendation": {
                "patterns": [
                    "What crop should I grow?",
                    "Which crop is best for my soil?",
                    "Suggest crops for my region",
                    "कौन सी फसल उगाऊं?",
                    "मेरी मिट्टी के लिए कौन सी फसल अच्छी है?",
                    "எந்த பயிர் வளர்க்க வேண்டும்?"
                ],
                "responses": [
                    "Based on your conditions, I recommend {crop}.",
                    "आपकी स्थिति के आधार पर, मैं {crop} की सिफारिश करता हूं।"
                ]
            },
            "fertilizer_advice": {
                "patterns": [
                    "How much fertilizer to use?",
                    "NPK ratio for crops",
                    "उर्वरक कितना देना चाहिए?",
                    "உரம் எவ்வளவு பயன்படுத்த வேண்டும்?"
                ],
                "responses": [
                    "For {crop}, use {fertilizer} at {rate} kg/ha.",
                    "{crop} के लिए {fertilizer} {rate} किग्रा/हेक्टेयर डालें।"
                ]
            },
            "disease_detection": {
                "patterns": [
                    "My plant has disease",
                    "Leaves are turning yellow",
                    "How to treat leaf spots?",
                    "पौधे में बीमारी है",
                    "இலைகள் மஞ்சளாகின்றன"
                ],
                "responses": [
                    "This appears to be {disease}. Treatment: {treatment}",
                    "यह {disease} लगता है। उपचार: {treatment}"
                ]
            }
        },
        "training_data": [],
        "metadata": {
            "languages": ["en", "hi", "ta"],
            "created": datetime.now().isoformat()
        }
    }
    
    # Generate training samples from intents
    for intent_name, intent_data in corpus['intents'].items():
        for pattern in intent_data['patterns']:
            corpus['training_data'].append({
                'text': pattern,
                'intent': intent_name
            })
    
    return corpus


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Build BGE-M3 embeddings for AgriSense')
    parser.add_argument('--corpus', type=str, default=None,
                        help='Path to corpus JSON file')
    parser.add_argument('--output-dir', type=str, default='./models/nlp',
                        help='Output directory')
    parser.add_argument('--generate-corpus', action='store_true',
                        help='Generate sample multilingual corpus')
    parser.add_argument('--test', action='store_true',
                        help='Test with sample queries')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🌐 AgriSense BGE-M3 Multilingual Embeddings")
    print("=" * 70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate corpus if requested
    if args.generate_corpus or not args.corpus:
        logger.info("Generating multilingual corpus...")
        corpus = generate_multilingual_corpus()
        
        corpus_path = output_dir / 'multilingual_corpus.json'
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Generated corpus: {corpus_path}")
        
        args.corpus = str(corpus_path)
    
    # Initialize knowledge base
    kb = AgriKnowledgeBase({
        'model_name': 'BAAI/bge-m3',
        'embedding_dim': 1024,
        'output_dir': args.output_dir
    })
    
    # Build index
    logger.info(f"Building index from: {args.corpus}")
    kb.build_index(args.corpus)
    
    # Save
    kb.save(args.output_dir)
    
    # Test queries
    if args.test:
        print("\n" + "=" * 70)
        print("🧪 Testing multilingual queries")
        print("=" * 70)
        
        test_queries = [
            "What fertilizer should I use for rice?",
            "धान के लिए कौन सा उर्वरक अच्छा है?",
            "நெல்லுக்கு எந்த உரம் நல்லது?",
            "How to control plant disease?",
            "पौधों की बीमारी कैसे रोकें?"
        ]
        
        for query in test_queries:
            result = kb.get_response(query)
            print(f"\n📝 Query: {query}")
            print(f"   Intent: {result['intent']} (conf: {result['confidence']:.2f})")
            print(f"   Response: {result['response'][:100]}...")
    
    print("\n" + "=" * 70)
    print("✅ BGE-M3 Embeddings Complete!")
    print("=" * 70)
    print(f"\nSaved to: {output_dir}")
    print(f"  - vector_store/")
    print(f"  - knowledge_base.json")
    print(f"  - multilingual_corpus.json")


if __name__ == '__main__':
    main()
