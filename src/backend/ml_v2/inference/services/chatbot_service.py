"""
Chatbot Service - RAG + Intent Classification with multilingual support.

Uses DistilBERT for intent and BGE-M3 for embeddings.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Structured chatbot response."""
    intent: str
    confidence: float
    response: str
    sources: Optional[List[Dict[str, Any]]] = None
    language: str = "en"


class ChatbotService:
    """
    Multilingual agricultural chatbot using RAG.
    
    Supports: English, Hindi, Tamil, Telugu, Kannada
    
    Usage:
        service = ChatbotService()
        response = service.query("मेरी फसल में पीला रोग है")
    """
    
    INTENTS = ["weather", "disease", "soil", "crop_recommendation", "pricing", "general"]
    
    def __init__(self):
        """Initialize chatbot components."""
        self.intent_model = None
        self.embeddings_model = None
        self.vector_store = None
        self._load_models()
    
    def _load_models(self):
        """Load intent classifier and embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            # BGE-M3 for multilingual embeddings
            self.embeddings_model = SentenceTransformer("BAAI/bge-m3")
            logger.info("Loaded BGE-M3 embeddings model")
        except Exception as e:
            logger.warning(f"Could not load embeddings: {e}")
    
    def query(
        self,
        user_input: str,
        language: Optional[str] = None
    ) -> ChatResponse:
        """
        Process user query with intent classification and RAG.
        
        Args:
            user_input: User's question (any supported language)
            language: Override language detection
            
        Returns:
            ChatResponse with answer and sources
        """
        # Detect language if not provided
        if not language:
            language = self._detect_language(user_input)
        
        # Classify intent
        intent, intent_conf = self._classify_intent(user_input)
        
        # Retrieve relevant context
        context = self._retrieve_context(user_input, intent)
        
        # Generate response
        response = self._generate_response(user_input, intent, context, language)
        
        return ChatResponse(
            intent=intent,
            confidence=intent_conf,
            response=response,
            sources=context,
            language=language
        )
    
    def _detect_language(self, text: str) -> str:
        """Detect text language."""
        try:
            from langdetect import detect
            lang = detect(text)
            # Map to supported languages
            lang_map = {"hi": "hi", "ta": "ta", "te": "te", "kn": "kn"}
            return lang_map.get(lang, "en")
        except:
            return "en"
    
    def _classify_intent(self, text: str) -> tuple:
        """Classify user intent."""
        # Simple keyword-based fallback (DistilBERT would replace this)
        text_lower = text.lower()
        
        intent_keywords = {
            "weather": ["weather", "rain", "temperature", "मौसम", "வானிலை"],
            "disease": ["disease", "pest", "blight", "रोग", "நோய்"],
            "soil": ["soil", "fertilizer", "npk", "मिट्टी", "மண்"],
            "crop_recommendation": ["recommend", "suggest", "grow", "उगाना", "வளர்க்க"],
            "pricing": ["price", "market", "sell", "कीमत", "விலை"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return intent, 0.8
        
        return "general", 0.5
    
    def _retrieve_context(
        self, 
        query: str, 
        intent: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using BGE-M3."""
        if not self.embeddings_model:
            return []
        
        # In production, this would query a vector store (FAISS/ChromaDB)
        # For now, return placeholder
        return [
            {"text": "Placeholder context", "score": 0.9}
        ]
    
    def _generate_response(
        self,
        query: str,
        intent: str,
        context: List[Dict],
        language: str
    ) -> str:
        """Generate response from context."""
        # Template-based response (could be LLM in production)
        templates = {
            "weather": "Based on current conditions, I recommend...",
            "disease": "The symptoms you described may indicate...",
            "soil": "For your soil type, consider...",
            "crop_recommendation": "Given your conditions, suitable crops are...",
            "pricing": "Current market prices show...",
            "general": "I can help you with crop recommendations, disease identification, and more."
        }
        
        return templates.get(intent, templates["general"])
