"""
AGRI-OS Demo Mode
==================
Precomputed pipeline for instant demonstrations.

Bundles precomputed embeddings, a mini FAISS index, and Governor decisions
so that /agrios/demo/run returns instant results without requiring GPU or
model loading.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .schemas import (
    ActionTemplate,
    AnalyzeResponse,
    AnomalyFlag,
    ConfidenceBand,
    DecisionAction,
    DemoRunResponse,
    GovernorDecision,
    VRAGResult,
)

logger = logging.getLogger("agrios.demo")

# Base path for demo data
DEMO_DATA_DIR = Path(__file__).parent / "demo_data"


# Precomputed demo scenarios
DEMO_SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "Tomato Late Blight (Diseased)",
        "image": "tomato_late_blight.jpg",
        "crop": "tomato",
        "disease": "Late_blight",
        "confidence": 0.92,
        "severity": "high",
        "action": "ACT",
        "anomaly": False,
    },
    {
        "name": "Rice Bacterial Leaf Blight (Diseased)",
        "image": "rice_blight.jpg",
        "crop": "rice",
        "disease": "Bacterial_leaf_blight",
        "confidence": 0.85,
        "severity": "medium",
        "action": "ACT",
        "anomaly": False,
    },
    {
        "name": "Corn Weed Infestation",
        "image": "corn_weed.jpg",
        "crop": "corn",
        "disease": "weed_competition",
        "confidence": 0.78,
        "severity": "medium",
        "action": "WAIT",
        "anomaly": False,
    },
    {
        "name": "Wheat Healthy Crop",
        "image": "wheat_healthy.jpg",
        "crop": "wheat",
        "disease": "healthy",
        "confidence": 0.95,
        "severity": "none",
        "action": "DO_NOTHING",
        "anomaly": False,
    },
    {
        "name": "Unknown Anomalous Input",
        "image": "anomaly_sample.jpg",
        "crop": "unknown",
        "disease": "unknown",
        "confidence": 0.35,
        "severity": "unknown",
        "action": "OBSERVE",
        "anomaly": True,
    },
]


def _generate_demo_embedding(seed: int = 42) -> np.ndarray:
    """Generate a deterministic 384-dim pseudo-embedding for demo."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(384).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


def _build_demo_response(scenario: Dict[str, Any], idx: int) -> AnalyzeResponse:
    """Build a precomputed AnalyzeResponse from a demo scenario."""
    conf = scenario["confidence"]
    is_anomaly = scenario["anomaly"]

    # Build confidence band
    if conf >= 0.8:
        band = ConfidenceBand(lower=conf - 0.08, median=conf, upper=min(conf + 0.05, 1.0))
    elif conf >= 0.6:
        band = ConfidenceBand(lower=conf - 0.15, median=conf, upper=min(conf + 0.1, 1.0))
    else:
        band = ConfidenceBand(lower=conf - 0.2, median=conf, upper=min(conf + 0.15, 1.0))

    # Build anomaly flag
    anomaly_flag = AnomalyFlag(
        is_anomaly=is_anomaly,
        anomaly_score=-0.3 if is_anomaly else 0.2,
        gate_action=DecisionAction.OBSERVE if is_anomaly else DecisionAction.ACT,
        reason=("OOD input detected" if is_anomaly else "Input within known distribution"),
    )

    # Build VRAG evidence
    vrag_evidence = []
    if not is_anomaly and scenario["disease"] != "healthy":
        vrag_evidence = [
            VRAGResult(
                doc_id=f"demo_match_{idx}_{j}",
                similarity_score=max(0.6, conf - j * 0.1),
                metadata={
                    "crop": scenario["crop"],
                    "disease": scenario["disease"],
                    "region": "demo",
                },
                evidence_text=f"Similar case: {scenario['crop']} with {scenario['disease']} " f"(demo match {j + 1})",
                source_image=None,
            )
            for j in range(3)
        ]

    # Build governor decision
    action = DecisionAction(scenario["action"])
    evidence = [
        f"Vision confidence: {conf:.2f}",
        f"Detected: {scenario['disease']}",
        f"Confidence band: [{band.lower:.2f}, {band.median:.2f}, {band.upper:.2f}]",
    ]
    if is_anomaly:
        evidence.append("⚠️ ANOMALY: OOD input detected")
        evidence.append("Action capped at OBSERVE due to anomaly gate")

    # Treatment template for ACT decisions
    treatment = None
    if action == DecisionAction.ACT:
        treatment = {
            "crop": scenario["crop"],
            "disease": scenario["disease"],
            "action_type": "ACT",
            "treatment": "Follow recommended treatment protocol",
            "dosage": "As per local agricultural guidelines",
            "timing": "Apply within 24 hours of detection",
            "safety_notes": "Wear protective equipment",
            "follow_up_days": 7,
        }

    governor = GovernorDecision(
        action=action,
        confidence_band=band,
        regret_score=0.05 if action == DecisionAction.ACT else 0.15,
        evidence=evidence,
        alternative_actions=[],
        treatment=treatment,
    )

    # Build explanation
    explanation_parts = [
        f"Demo scenario: {scenario['name']}.",
        f"The analysis detected {scenario['disease']} on {scenario['crop']} with {conf:.0%} confidence.",
    ]
    if action == DecisionAction.ACT:
        explanation_parts.append("Immediate treatment is recommended.")
    elif action == DecisionAction.WAIT:
        explanation_parts.append("Monitoring is recommended before taking action.")
    elif action == DecisionAction.OBSERVE:
        explanation_parts.append("Continue observing — no action should be taken yet.")
    else:
        explanation_parts.append("The crop appears healthy — no intervention needed.")

    return AnalyzeResponse(
        detection={
            "disease": scenario["disease"],
            "crop": scenario["crop"],
            "confidence": conf,
            "severity": scenario["severity"],
            "source": "demo_precomputed",
        },
        governor_decision=governor,
        evidence=vrag_evidence,
        explanation=" ".join(explanation_parts),
        actions=(
            [
                ActionTemplate(
                    crop=scenario["crop"],
                    disease=scenario["disease"],
                    action_type=action.value,
                    treatment="Demo treatment protocol",
                    dosage="As per guidelines",
                    timing="See recommendation",
                    safety_notes="Demo mode — consult expert for real cases",
                    follow_up_days=7,
                )
            ]
            if action == DecisionAction.ACT
            else []
        ),
        anomaly_flag=anomaly_flag,
        pipeline_version="1.0.0-demo",
    )


def run_demo_pipeline() -> DemoRunResponse:
    """
    Run the full demo pipeline with precomputed data.
    Returns instantly without requiring any model loading.
    """
    start = time.time()

    results = []
    for idx, scenario in enumerate(DEMO_SCENARIOS):
        response = _build_demo_response(scenario, idx)
        results.append(response)

    elapsed_ms = (time.time() - start) * 1000

    return DemoRunResponse(
        results=results,
        demo_images_used=len(DEMO_SCENARIOS),
        total_time_ms=round(elapsed_ms, 2),
        message=f"Demo pipeline completed: {len(DEMO_SCENARIOS)} scenarios processed in {elapsed_ms:.0f}ms",
    )


def ensure_demo_data() -> None:
    """Create demo_data directory structure if it doesn't exist."""
    dirs = [
        DEMO_DATA_DIR / "demo_images",
        DEMO_DATA_DIR / "demo_embeddings",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Generate demo embeddings
    for idx, scenario in enumerate(DEMO_SCENARIOS):
        emb = _generate_demo_embedding(seed=42 + idx)
        emb_path = DEMO_DATA_DIR / "demo_embeddings" / f"{scenario['image'].replace('.jpg', '.npy')}"
        np.save(str(emb_path), emb)

    # Save demo results
    demo_results = run_demo_pipeline()
    results_path = DEMO_DATA_DIR / "demo_results.json"
    with open(results_path, "w") as f:
        json.dump(demo_results.model_dump(), f, indent=2, default=str)

    logger.info("Demo data ensured at %s", DEMO_DATA_DIR)
