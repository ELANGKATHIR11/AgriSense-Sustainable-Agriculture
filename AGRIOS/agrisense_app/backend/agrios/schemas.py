"""
AGRI-OS Shared Pydantic Schemas
================================
All data models used across the AGRI-OS pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DecisionAction(str, Enum):
    """Possible actions the Decision Governor can recommend."""

    ACT = "ACT"
    WAIT = "WAIT"
    OBSERVE = "OBSERVE"
    DO_NOTHING = "DO_NOTHING"


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


class VisionResult(BaseModel):
    """Result from the DeiT embedding + SCOLD + Triplet pipeline."""

    embedding: Optional[List[float]] = Field(None, description="384-dim DeiT CLS token embedding")
    top_k_matches: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top-K nearest disease/crop matches from VRAG",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Pipeline confidence")
    disease_name: Optional[str] = Field(None, description="Predicted disease name")
    crop_type: Optional[str] = Field(None, description="Detected or provided crop type")
    severity: Optional[str] = Field(None, description="Estimated severity level")
    scold_projection: Optional[List[float]] = Field(None, description="128-dim SCOLD projected embedding")


# ---------------------------------------------------------------------------
# Decision Governor
# ---------------------------------------------------------------------------


class ConfidenceBand(BaseModel):
    """Bootstrap percentile confidence band."""

    lower: float = Field(..., ge=0.0, le=1.0)
    median: float = Field(..., ge=0.0, le=1.0)
    upper: float = Field(..., ge=0.0, le=1.0)


class GovernorDecision(BaseModel):
    """Output of the Decision Governor."""

    action: DecisionAction = Field(..., description="Recommended action")
    confidence_band: ConfidenceBand = Field(..., description="Bootstrap confidence interval")
    regret_score: float = Field(..., ge=0.0, description="Minimax regret score for chosen action")
    evidence: List[str] = Field(default_factory=list, description="Evidence chain supporting the decision")
    alternative_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ranked alternative actions with regret scores",
    )
    treatment: Optional[Dict[str, Any]] = Field(None, description="Structured treatment guidance when action=ACT")


class ActionTemplate(BaseModel):
    """Treatment/action template for a specific crop-disease combination."""

    crop: str
    disease: str
    action_type: str
    treatment: str
    dosage: str
    timing: str
    safety_notes: str
    follow_up_days: int = 7


# ---------------------------------------------------------------------------
# VRAG
# ---------------------------------------------------------------------------


class VRAGResult(BaseModel):
    """Single retrieval result from the FAISS-backed VRAG."""

    doc_id: str = Field(..., description="Unique document/embedding identifier")
    similarity_score: float = Field(..., description="Cosine / IP similarity")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Associated metadata (crop, disease, region, severity, outcome)",
    )
    evidence_text: str = Field("", description="Human-readable evidence summary")
    source_image: Optional[str] = Field(None, description="Path to the source image if available")


class VRAGQueryRequest(BaseModel):
    """Request to query the VRAG index."""

    embedding: List[float] = Field(..., description="Query embedding (384-dim)")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    crop_filter: Optional[str] = Field(None, description="Filter by crop type")
    region_filter: Optional[str] = Field(None, description="Filter by region")


class VRAGQueryResponse(BaseModel):
    """Response from the VRAG index query."""

    results: List[VRAGResult] = Field(default_factory=list)
    query_time_ms: float = Field(0.0, description="Query latency in milliseconds")
    index_size: int = Field(0, description="Total entries in the FAISS index")


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------


class AnomalyFlag(BaseModel):
    """Result from the Isolation Forest anomaly gate."""

    is_anomaly: bool = Field(..., description="True if embedding is OOD")
    anomaly_score: float = Field(..., description="Isolation Forest anomaly score (-1 to 0 typical)")
    gate_action: DecisionAction = Field(..., description="Forced action when anomaly detected")
    reason: str = Field("", description="Human-readable reason for the flag")


# ---------------------------------------------------------------------------
# Full Pipeline Request / Response
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Full pipeline analysis request."""

    image: str = Field(..., description="Base64-encoded image data")
    crop_type: Optional[str] = Field(None, description="Known crop type")
    sensor_data: Optional[Dict[str, Any]] = Field(None, description="Optional sensor readings")
    language: str = Field("en", description="Response language code")


class AnalyzeResponse(BaseModel):
    """Full pipeline analysis response."""

    detection: Dict[str, Any] = Field(default_factory=dict, description="Raw detection results")
    governor_decision: Optional[GovernorDecision] = Field(None, description="Decision Governor output")
    evidence: List[VRAGResult] = Field(default_factory=list, description="VRAG evidence chain")
    explanation: str = Field("", description="GenAI natural-language explanation")
    actions: List[ActionTemplate] = Field(default_factory=list, description="Applicable action templates")
    anomaly_flag: Optional[AnomalyFlag] = Field(None, description="Anomaly gate result")
    pipeline_version: str = Field("1.0.0", description="AGRI-OS pipeline version")


class DecisionEvaluateRequest(BaseModel):
    """Manual sensor-only decision evaluation request."""

    sensor_data: Dict[str, Any] = Field(..., description="Sensor readings")
    crop_type: str = Field(..., description="Crop type")
    crop_context: Optional[Dict[str, Any]] = Field(None, description="Additional crop context")


class DemoRunResponse(BaseModel):
    """Demo pipeline response with precomputed data."""

    results: List[AnalyzeResponse] = Field(default_factory=list)
    demo_images_used: int = Field(0)
    total_time_ms: float = Field(0.0)
    message: str = Field("Demo pipeline completed successfully")


class HealthResponse(BaseModel):
    """Pipeline health check response."""

    status: str = Field("ok")
    components: Dict[str, bool] = Field(default_factory=dict)
    version: str = Field("1.0.0")
    vram_usage_mb: Optional[float] = Field(None)
    index_size: Optional[int] = Field(None)


# ---------------------------------------------------------------------------
# Feedback Loop (Patent 07)
# ---------------------------------------------------------------------------


class OutcomeReport(BaseModel):
    """Farmer-reported outcome of a recommended action."""

    crop_type: str = Field(..., description="Crop type for this outcome")
    disease: str = Field("", description="Disease that was diagnosed")
    decision_action: DecisionAction = Field(..., description="Action that was recommended")
    outcome: str = Field(
        ...,
        description=("Outcome: recovered / worsened / no_change " "/ false_alarm / missed_detection"),
    )
    notes: str = Field("", description="Optional farmer notes")
    image_base64: Optional[str] = Field(
        None,
        description=("Optional follow-up image (base64) to add " "to VRAG index with outcome metadata"),
    )
    yield_impact_pct: Optional[float] = Field(
        None,
        description="Estimated yield impact percentage",
    )


class OutcomeResponse(BaseModel):
    """Response after recording an outcome."""

    recorded: bool = Field(True)
    feedback_count: int = Field(0)
    vrag_updated: bool = Field(False)
    anomaly_retrained: bool = Field(False)
    weights_adjusted: bool = Field(False)
    message: str = Field("Outcome recorded")
