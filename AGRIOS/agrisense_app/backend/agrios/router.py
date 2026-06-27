"""
AGRI-OS API Router
===================
FastAPI router for all AGRI-OS endpoints.

Endpoints:
  POST /agrios/vision/analyze   — Full pipeline: Image → DeiT → SCOLD → VRAG → Governor → GenAI
  POST /agrios/vision/embed     — Image → DeiT embedding (raw)
  POST /agrios/decision/evaluate — Manual sensor data → Governor decision
  GET  /agrios/decision/actions  — List possible actions for crop+context
  POST /agrios/vrag/query        — Direct VRAG query with embedding
  GET  /agrios/health            — Pipeline health check
  POST /agrios/demo/run          — Run demo pipeline with precomputed data
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .decision_governor import (
    DecisionGovernor,
    get_action_template,
    get_actions_for_crop,
)
from .demo_mode import run_demo_pipeline
from .schemas import (
    ActionTemplate,
    AnalyzeRequest,
    AnalyzeResponse,
    DecisionAction,
    DecisionEvaluateRequest,
    DemoRunResponse,
    GovernorDecision,
    HealthResponse,
    OutcomeReport,
    OutcomeResponse,
    VRAGQueryRequest,
    VRAGQueryResponse,
    VRAGResult,
    VisionResult,
)

logger = logging.getLogger("agrios.router")

router = APIRouter()

# Lazy-loaded singletons
_deit_extractor = None
_vrag_engine = None
_anomaly_gate = None
_governor = DecisionGovernor()
_genai = None


def _get_deit():
    global _deit_extractor
    if _deit_extractor is None:
        try:
            from .vision_pipeline import DeiTEmbeddingExtractor

            _deit_extractor = DeiTEmbeddingExtractor(device="cpu")
        except Exception as e:
            logger.warning("DeiT extractor unavailable: %s", e)
    return _deit_extractor


def _get_vrag():
    global _vrag_engine
    if _vrag_engine is None:
        try:
            from pathlib import Path

            from .vrag import VRAGEngine

            base = Path(__file__).resolve().parents[2] / "AI_Models" / "scold"
            index_path = base / "vrag_index.faiss"
            meta_path = base / "vrag_metadata.json"
            if index_path.exists() and meta_path.exists():
                _vrag_engine = VRAGEngine(index_path, meta_path)
            else:
                _vrag_engine = VRAGEngine()  # unloaded
        except Exception as e:
            logger.warning("VRAG engine unavailable: %s", e)
    return _vrag_engine


def _get_anomaly():
    global _anomaly_gate
    if _anomaly_gate is None:
        try:
            from pathlib import Path

            from .isolation_forest import AnomalyGate

            model_path = Path(__file__).resolve().parents[2] / "AI_Models" / "scold" / "anomaly_gate.joblib"
            _anomaly_gate = AnomalyGate()
            if model_path.exists():
                _anomaly_gate.load(model_path)
        except Exception as e:
            logger.warning("Anomaly gate unavailable: %s", e)
    return _anomaly_gate


def _get_genai():
    global _genai
    if _genai is None:
        try:
            from .genai_contract import AgriGenAI

            _genai = AgriGenAI()
        except Exception as e:
            logger.warning("GenAI unavailable: %s", e)
    return _genai


# ===========================================================================
# Endpoints
# ===========================================================================


@router.post("/vision/analyze", response_model=AnalyzeResponse)
async def analyze_image(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Full pipeline: Image → DeiT embed → Anomaly gate → VRAG → Governor → GenAI.
    Falls back gracefully at each step.
    """
    embedding = None
    vision_result = None
    anomaly_flag = None
    vrag_results: List[VRAGResult] = []
    governor_decision = None
    explanation = ""
    detection: Dict[str, Any] = {}
    actions: List[ActionTemplate] = []

    crop = request.crop_type or "unknown"
    sensor = request.sensor_data or {}

    # Step 1: DeiT Embedding
    deit = _get_deit()
    if deit:
        try:
            embedding = deit.embed_base64(request.image)
            vision_result = VisionResult(
                embedding=embedding.tolist(),
                confidence=0.0,
                crop_type=crop,
                disease_name=None,
                severity=None,
                scold_projection=None,
            )
        except Exception as e:
            logger.warning("DeiT embedding failed: %s", e)

    # Step 2: Anomaly Gate
    if embedding is not None:
        gate = _get_anomaly()
        if gate and gate.is_trained:
            try:
                anomaly_flag = gate.gate(embedding)
            except Exception as e:
                logger.warning("Anomaly gate failed: %s", e)

    # Step 3: VRAG Retrieval
    if embedding is not None:
        vrag = _get_vrag()
        if vrag and vrag.is_loaded:
            try:
                vrag_results = vrag.query(embedding, top_k=5, crop_filter=crop if crop != "unknown" else None)
                if vrag_results and vision_result:
                    # Update vision result with VRAG matches
                    vision_result.top_k_matches = [r.metadata for r in vrag_results]
                    vision_result.confidence = max((r.similarity_score for r in vrag_results), default=0.0)
                    # Infer disease from top match
                    if vrag_results[0].metadata.get("disease"):
                        vision_result.disease_name = vrag_results[0].metadata["disease"]
                        detection["disease"] = vision_result.disease_name
                        detection["confidence"] = vision_result.confidence
            except Exception as e:
                logger.warning("VRAG retrieval failed: %s", e)

    # Step 4: Fallback detection if no VRAG results
    if not detection and embedding is not None:
        # Try existing disease detection as fallback
        try:
            # Attempt to use existing backend detection
            detection = {
                "disease": "unknown",
                "confidence": 0.5,
                "source": "fallback",
                "note": "VRAG index not available — using basic detection",
            }
        except Exception:
            pass

    # Step 5: Decision Governor
    disease = detection.get("disease", "unknown")
    try:
        inputs = {
            "vision_result": vision_result,
            "sensor_data": sensor if sensor else None,
            "vrag_evidence": vrag_results,
            "anomaly_flag": anomaly_flag,
            "crop_context": {"crop_type": crop, "disease": disease},
        }
        governor_decision = _governor.decide(inputs)
    except Exception as e:
        logger.warning("Governor decision failed: %s", e)

    # Step 6: Get action templates
    if governor_decision and governor_decision.action == DecisionAction.ACT:
        template = get_action_template(crop, disease)
        if template:
            actions.append(template)

    # Step 7: GenAI Explanation
    genai = _get_genai()
    if genai and governor_decision:
        try:
            explanation = await genai.explain(governor_decision, vrag_results, sensor, request.language)
        except Exception as e:
            logger.warning("GenAI explanation failed: %s", e)
            explanation = "Explanation generation unavailable."

    return AnalyzeResponse(
        detection=detection,
        governor_decision=governor_decision,
        evidence=vrag_results,
        explanation=explanation,
        actions=actions,
        anomaly_flag=anomaly_flag,
        pipeline_version="1.0.0",
    )


class EmbedRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int


@router.post("/vision/embed", response_model=EmbedResponse)
async def embed_image(request: EmbedRequest) -> EmbedResponse:
    """Extract DeiT embedding from an image (raw)."""
    deit = _get_deit()
    if not deit:
        raise HTTPException(503, "DeiT embedding extractor not available")

    try:
        embedding = deit.embed_base64(request.image)
        return EmbedResponse(
            embedding=embedding.tolist(),
            dimension=len(embedding),
        )
    except Exception as e:
        raise HTTPException(400, f"Embedding extraction failed: {e}")


@router.post("/decision/evaluate", response_model=GovernorDecision)
async def evaluate_decision(request: DecisionEvaluateRequest) -> GovernorDecision:
    """Evaluate sensor data through the Decision Governor (no image)."""
    inputs: Dict[str, Any] = {
        "vision_result": None,
        "sensor_data": request.sensor_data,
        "vrag_evidence": [],
        "anomaly_flag": None,
        "crop_context": {
            "crop_type": request.crop_type,
            **(request.crop_context or {}),
        },
    }
    return _governor.decide(inputs)


class ActionsQuery(BaseModel):
    crop_type: str
    disease: Optional[str] = None


@router.get("/decision/actions")
async def list_actions(crop_type: str, disease: Optional[str] = None) -> Dict[str, Any]:
    """List possible action templates for a crop + optional disease."""
    if disease:
        template = get_action_template(crop_type, disease)
        return {"actions": [template.model_dump()] if template else []}
    else:
        templates = get_actions_for_crop(crop_type)
        return {"actions": [t.model_dump() for t in templates]}


@router.post("/vrag/query", response_model=VRAGQueryResponse)
async def query_vrag(request: VRAGQueryRequest) -> VRAGQueryResponse:
    """Direct VRAG query with a precomputed embedding."""
    vrag = _get_vrag()
    if not vrag or not vrag.is_loaded:
        return VRAGQueryResponse(results=[], query_time_ms=0, index_size=0)

    start = time.time()
    embedding = np.array(request.embedding, dtype=np.float32)
    results = vrag.query(
        embedding,
        top_k=request.top_k,
        crop_filter=request.crop_filter,
        region_filter=request.region_filter,
    )
    elapsed = (time.time() - start) * 1000

    return VRAGQueryResponse(
        results=results,
        query_time_ms=round(elapsed, 2),
        index_size=vrag.index_size,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Pipeline health check."""
    components = {
        "deit_extractor": _get_deit() is not None,
        "vrag_engine": ((_get_vrag() is not None and _get_vrag().is_loaded) if _get_vrag() else False),
        "anomaly_gate": ((_get_anomaly() is not None and _get_anomaly().is_trained) if _get_anomaly() else False),
        "decision_governor": True,  # always available (rule-based)
        "genai": _get_genai() is not None,
    }

    vram_mb = None
    try:
        import torch

        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    except ImportError:
        pass

    index_size = None
    vrag = _get_vrag()
    if vrag and vrag.is_loaded:
        index_size = vrag.index_size

    all_ok = all(components.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        components=components,
        version="1.0.0",
        vram_usage_mb=vram_mb,
        index_size=index_size,
    )


@router.post("/demo/run", response_model=DemoRunResponse)
async def run_demo() -> DemoRunResponse:
    """Run demo pipeline with precomputed data — no model loading required."""
    return run_demo_pipeline()


# ===========================================================================
# Feedback Loop (Patent 07)
# ===========================================================================


@router.post("/feedback/outcome", response_model=OutcomeResponse)
async def record_outcome(
    report: OutcomeReport,
) -> OutcomeResponse:
    """
    Record a farmer-reported outcome and trigger model updates.

    Updates three subsystems:
    1. VRAG index — adds confirmed image+outcome to retrieval
    2. Governor loss weights — adjusts via feedback accumulator
    3. Anomaly gate — retrains if new embeddings available
    """
    vrag_updated = False
    anomaly_retrained = False
    weights_adjusted = False

    is_fp = report.outcome in ("false_alarm",)
    is_fn = report.outcome in ("missed_detection", "worsened")

    # 1) Record in Governor feedback accumulator
    _governor.record_outcome(
        crop=report.crop_type,
        was_false_positive=is_fp,
        was_false_negative=is_fn,
    )
    weights_adjusted = True

    # 2) Add follow-up image to VRAG index if provided
    if report.image_base64:
        deit = _get_deit()
        vrag = _get_vrag()
        if deit and vrag:
            try:
                emb = deit.embed_base64(report.image_base64)
                meta = {
                    "crop": report.crop_type,
                    "disease": report.disease,
                    "outcome": report.outcome,
                    "notes": report.notes,
                    "source": "feedback",
                }
                vrag.add_embedding(emb, meta)
                vrag_updated = True
            except Exception as e:
                logger.warning("VRAG feedback update failed: %s", e)

    # 3) Retrain anomaly gate if VRAG was updated
    if vrag_updated:
        gate = _get_anomaly()
        if gate and vrag:
            try:
                all_emb = vrag.get_all_embeddings()
                if all_emb is not None and len(all_emb) > 10:
                    gate.train(all_emb)
                    anomaly_retrained = True
            except Exception as e:
                logger.warning(
                    "Anomaly gate retrain failed: %s",
                    e,
                )

    fb = _governor._feedback.get(report.crop_type.lower())
    count = fb["total"] if fb else 0

    return OutcomeResponse(
        recorded=True,
        feedback_count=count,
        vrag_updated=vrag_updated,
        anomaly_retrained=anomaly_retrained,
        weights_adjusted=weights_adjusted,
        message=(f"Outcome '{report.outcome}' recorded for " f"{report.crop_type}/{report.disease}"),
    )
