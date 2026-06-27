"""
AGRI-OS Edge-Autonomous System
===============================

Core modules for the AGRI-OS intelligent agricultural decision system:
- Vision Pipeline: DeiT embedding extraction, SCOLD training, Triplet networks
- VRAG: FAISS-backed Visual Retrieval-Augmented Generation
- Decision Governor: Regret-based action gating with confidence bands
- GenAI Contract: RAG-only LLM wrapper with strict evidence grounding
- Isolation Forest: Anomaly detection and OOD gating
- Demo Mode: Precomputed pipeline for instant demonstrations
"""

__version__ = "1.0.0"
__all__ = [
    "vision_pipeline",
    "vrag",
    "decision_governor",
    "genai_contract",
    "isolation_forest",
    "demo_mode",
    "router",
    "schemas",
]
