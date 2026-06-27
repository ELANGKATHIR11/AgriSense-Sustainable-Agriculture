"""
RAG Adapter — Bridge between existing chatbot routes and AGRI-OS GenAI system.

When vision/sensor context is available, routes through the Governor+VRAG
pipeline for evidence-grounded responses. Otherwise falls through to the
existing ChromaDB-based chatbot.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("agrios.rag_adapter")

# Lazy imports to avoid circular dependencies
_agri_genai = None


def _get_genai():
    global _agri_genai
    if _agri_genai is None:
        try:
            from agrios.genai_contract import AgriGenAI

            _agri_genai = AgriGenAI()
        except Exception as e:
            logger.debug("AgriGenAI not available: %s", e)
    return _agri_genai


async def enhanced_chatbot_answer(
    question: str,
    vision_context: Optional[Dict[str, Any]] = None,
    sensor_context: Optional[Dict[str, Any]] = None,
    crop_type: Optional[str] = None,
    language: str = "en",
) -> Optional[Dict[str, Any]]:
    """
    Try to answer a chatbot question using the AGRI-OS Governor+VRAG pipeline.

    Returns None if the enhanced pipeline is unavailable or if the question
    doesn't have vision/sensor context — caller should fall back to standard
    chatbot flow.

    Parameters
    ----------
    question : user's question
    vision_context : optional dict with 'image' (base64) or 'embedding'
    sensor_context : optional sensor readings
    crop_type : optional crop type
    language : response language

    Returns
    -------
    dict with 'answer', 'enhanced', 'confidence', 'evidence' or None
    """
    if not vision_context and not sensor_context:
        return None  # No enhanced context → use standard chatbot

    genai = _get_genai()
    if genai is None:
        return None

    try:
        from agrios.decision_governor import DecisionGovernor
        from agrios.schemas import VisionResult

        governor = DecisionGovernor()

        # Build governor inputs
        inputs: Dict[str, Any] = {
            "vision_result": None,
            "sensor_data": sensor_context,
            "vrag_evidence": [],
            "anomaly_flag": None,
            "crop_context": {"crop_type": crop_type or "unknown"},
        }

        # If we have vision context with an embedding, query VRAG
        vrag_evidence = []
        if vision_context and "embedding" in vision_context:
            try:
                import numpy as np
                from agrios.vrag import VRAGEngine
                from pathlib import Path

                base = Path(__file__).resolve().parent / "AI_Models" / "scold"
                index_path = base / "vrag_index.faiss"
                meta_path = base / "vrag_metadata.json"

                if index_path.exists() and meta_path.exists():
                    vrag = VRAGEngine(index_path, meta_path)
                    embedding = np.array(vision_context["embedding"], dtype=np.float32)
                    vrag_evidence = vrag.query(embedding, top_k=5)
                    inputs["vrag_evidence"] = vrag_evidence

                    # Create vision result with all required fields
                    max_confidence = max((r.similarity_score for r in vrag_evidence), default=0.5) if vrag_evidence else 0.5
                    inputs["vision_result"] = VisionResult(
                        embedding=vision_context["embedding"],
                        confidence=max_confidence,
                        crop_type=crop_type,
                        disease_name=vision_context.get("disease_name", "unknown"),
                        severity=vision_context.get("severity", "unknown"),
                        scold_projection=vision_context.get("scold_projection", {}),
                    )
            except Exception as e:
                logger.debug("VRAG query in rag_adapter failed: %s", e)

        # Get governor decision
        decision = governor.decide(inputs)

        # Generate explanation
        explanation = await genai.explain(decision, vrag_evidence, sensor_context or {}, language)

        return {
            "answer": explanation,
            "enhanced": True,
            "confidence": decision.confidence_band.median,
            "action": decision.action.value,
            "evidence": [r.evidence_text for r in vrag_evidence[:3]],
            "governor_decision": decision.model_dump(),
        }

    except Exception as e:
        logger.warning("Enhanced chatbot answer failed: %s", e)
        return None
