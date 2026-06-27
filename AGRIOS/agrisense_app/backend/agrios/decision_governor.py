"""
AGRI-OS Decision Governor
==========================
Regret-based action gating with confidence bands.

The Decision Governor is the central arbiter of all agricultural actions.
It fuses vision results, sensor data, VRAG evidence, and anomaly flags
into a single GovernorDecision using minimax regret ranking.

Decision Logic:
  1. Check anomaly gate → if flagged, cap at OBSERVE
  2. Compute confidence band (bootstrap percentile)
  3. Apply thresholds:
     - lower_bound < 0.4 → DO_NOTHING
     - lower_bound < 0.6 → OBSERVE
     - lower_bound < 0.8 → WAIT
     - lower_bound ≥ 0.8 AND no anomaly → ACT
  4. Compute regret: max(0, expected_loss(action) - expected_loss(best_alt))
  5. Annotate with evidence chain + optional treatment template
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .schemas import (
    ActionTemplate,
    AnomalyFlag,
    ConfidenceBand,
    DecisionAction,
    GovernorDecision,
    VRAGResult,
    VisionResult,
)

logger = logging.getLogger("agrios.governor")


# ======================== Crop-Specific Loss Weights =======================

# Default loss weights: how costly a wrong action is per crop category
# Higher = more conservative (favors WAIT/OBSERVE over ACT)
DEFAULT_LOSS_WEIGHTS: Dict[str, Dict[str, float]] = {
    "default": {
        "false_positive_act": 0.7,  # acting when shouldn't
        "missed_detection": 0.8,  # not acting when should
        "unnecessary_wait": 0.2,  # waiting when could act
        "observe_cost": 0.1,  # cost of observation
    },
    "rice": {
        "false_positive_act": 0.6,
        "missed_detection": 0.9,
        "unnecessary_wait": 0.15,
        "observe_cost": 0.1,
    },
    "tomato": {
        "false_positive_act": 0.8,
        "missed_detection": 0.7,
        "unnecessary_wait": 0.25,
        "observe_cost": 0.1,
    },
    "wheat": {
        "false_positive_act": 0.5,
        "missed_detection": 0.85,
        "unnecessary_wait": 0.2,
        "observe_cost": 0.1,
    },
    "potato": {
        "false_positive_act": 0.75,
        "missed_detection": 0.8,
        "unnecessary_wait": 0.2,
        "observe_cost": 0.1,
    },
}


# ========================= Action Templates ================================

# Map (crop, disease, action_type) → structured guidance
_ACTION_TEMPLATES: List[Dict[str, Any]] = [
    {
        "crop": "tomato",
        "disease": "Late_blight",
        "action_type": "ACT",
        "treatment": "Apply copper-based fungicide (Bordeaux mixture)",
        "dosage": "2-3 kg/ha copper hydroxide",
        "timing": "Apply within 24 hours of detection, repeat every 7-10 days",
        "safety_notes": "Wear protective equipment. Do not apply within 7 days of harvest.",
        "follow_up_days": 7,
    },
    {
        "crop": "tomato",
        "disease": "Early_blight",
        "action_type": "ACT",
        "treatment": "Apply chlorothalonil or mancozeb fungicide",
        "dosage": "1.5-2 kg/ha",
        "timing": "Apply at first sign of symptoms, repeat every 7-14 days",
        "safety_notes": "Avoid application during high wind. Use respirator.",
        "follow_up_days": 10,
    },
    {
        "crop": "potato",
        "disease": "Late_blight",
        "action_type": "ACT",
        "treatment": "Apply metalaxyl or phosphorous acid systemic fungicide",
        "dosage": "2.5 kg/ha",
        "timing": "Preventive application before infection expected",
        "safety_notes": "Rotate fungicide modes of action to prevent resistance.",
        "follow_up_days": 7,
    },
    {
        "crop": "rice",
        "disease": "Bacterial_leaf_blight",
        "action_type": "ACT",
        "treatment": "Drain field, apply streptocycline (antibiotic)",
        "dosage": "0.5 g/L foliar spray",
        "timing": "Apply during early infection stage, morning hours",
        "safety_notes": "Avoid overuse to prevent antibiotic resistance.",
        "follow_up_days": 14,
    },
    {
        "crop": "wheat",
        "disease": "Leaf_rust",
        "action_type": "ACT",
        "treatment": "Apply triazole-based fungicide (propiconazole)",
        "dosage": "0.5 L/ha",
        "timing": "Apply at flag leaf emergence if disease present",
        "safety_notes": "Single application usually sufficient for wheat rust.",
        "follow_up_days": 21,
    },
    {
        "crop": "corn",
        "disease": "Northern_Leaf_Blight",
        "action_type": "ACT",
        "treatment": "Apply strobilurin + triazole fungicide mix",
        "dosage": "1 L/ha",
        "timing": "Apply before tasseling when lesions first appear",
        "safety_notes": "Check product label for pre-harvest interval.",
        "follow_up_days": 14,
    },
    {
        "crop": "grape",
        "disease": "Black_rot",
        "action_type": "ACT",
        "treatment": "Apply myclobutanil or mancozeb",
        "dosage": "Label rate, typically 2-3 lb/acre mancozeb",
        "timing": "Begin at bud break, continue through bloom",
        "safety_notes": "Remove mummified berries to reduce inoculum.",
        "follow_up_days": 10,
    },
    {
        "crop": "apple",
        "disease": "Apple_scab",
        "action_type": "ACT",
        "treatment": "Apply captan or dodine fungicide",
        "dosage": "Label rate",
        "timing": "Begin at green tip, continue through petal fall",
        "safety_notes": "Avoid application in rain. Monitor for resistance.",
        "follow_up_days": 7,
    },
]


def get_action_template(crop: str, disease: str, action_type: str = "ACT") -> Optional[ActionTemplate]:
    """Look up an action template for a crop-disease-action combination."""
    for t in _ACTION_TEMPLATES:
        if (
            t["crop"].lower() == crop.lower()
            and t["disease"].lower() == disease.lower()
            and t["action_type"] == action_type
        ):
            return ActionTemplate(**t)
    # Generic fallback
    return ActionTemplate(
        crop=crop,
        disease=disease,
        action_type=action_type,
        treatment="Consult local agricultural extension for specific treatment",
        dosage="As per local guidelines",
        timing="Apply as soon as possible after confirmation",
        safety_notes="Always wear protective equipment. Follow label instructions.",
        follow_up_days=7,
    )


def get_actions_for_crop(crop: str) -> List[ActionTemplate]:
    """List all known action templates for a crop type."""
    results = []
    for t in _ACTION_TEMPLATES:
        if t["crop"].lower() == crop.lower():
            results.append(ActionTemplate(**t))
    return results


# ========================= Decision Governor ===============================


# ================== Growth-Stage / Season Multipliers ==================

# Multipliers adjust loss weights by growth stage.
# Higher missed_detection multiplier at critical stages.
GROWTH_STAGE_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "seedling": {
        "false_positive_act": 1.2,
        "missed_detection": 0.6,
        "unnecessary_wait": 0.8,
        "observe_cost": 1.0,
    },
    "vegetative": {
        "false_positive_act": 1.0,
        "missed_detection": 0.8,
        "unnecessary_wait": 1.0,
        "observe_cost": 1.0,
    },
    "flowering": {
        "false_positive_act": 0.9,
        "missed_detection": 1.3,
        "unnecessary_wait": 1.2,
        "observe_cost": 0.8,
    },
    "fruiting": {
        "false_positive_act": 0.8,
        "missed_detection": 1.4,
        "unnecessary_wait": 1.3,
        "observe_cost": 0.7,
    },
    "mature": {
        "false_positive_act": 1.1,
        "missed_detection": 1.0,
        "unnecessary_wait": 0.9,
        "observe_cost": 1.0,
    },
}

# Seasonal multipliers — monsoon increases disease risk.
SEASON_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "monsoon": {
        "false_positive_act": 0.8,
        "missed_detection": 1.4,
        "unnecessary_wait": 1.3,
        "observe_cost": 0.8,
    },
    "dry": {
        "false_positive_act": 1.1,
        "missed_detection": 0.7,
        "unnecessary_wait": 0.8,
        "observe_cost": 1.0,
    },
    "winter": {
        "false_positive_act": 1.0,
        "missed_detection": 0.9,
        "unnecessary_wait": 0.9,
        "observe_cost": 1.0,
    },
    "default": {
        "false_positive_act": 1.0,
        "missed_detection": 1.0,
        "unnecessary_wait": 1.0,
        "observe_cost": 1.0,
    },
}


class DecisionGovernor:
    """
    Central decision arbiter for the AGRI-OS pipeline.

    Fuses multiple signals (vision, sensor, VRAG, anomaly)
    into a single decision using minimax regret ranking and
    bootstrap confidence bands.

    Supports dynamic loss weight adjustment via:
    - growth_stage (seedling / vegetative / flowering / fruiting / mature)
    - season (monsoon / dry / winter)
    - feedback history (false-positive / false-negative rates)
    """

    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._custom_weights = loss_weights
        # Feedback accumulators per crop
        self._feedback: Dict[str, Dict[str, int]] = {}

    # ---------- feedback loop ----------

    def record_outcome(
        self,
        crop: str,
        was_false_positive: bool = False,
        was_false_negative: bool = False,
    ) -> None:
        """Record a decision outcome for dynamic weight adjustment."""
        key = crop.lower()
        if key not in self._feedback:
            self._feedback[key] = {
                "total": 0,
                "false_positives": 0,
                "false_negatives": 0,
            }
        self._feedback[key]["total"] += 1
        if was_false_positive:
            self._feedback[key]["false_positives"] += 1
        if was_false_negative:
            self._feedback[key]["false_negatives"] += 1

    def _feedback_multipliers(
        self,
        crop: str,
    ) -> Dict[str, float]:
        """Derive multipliers from accumulated feedback."""
        fb = self._feedback.get(crop.lower())
        if not fb or fb["total"] < 5:
            return {
                "false_positive_act": 1.0,
                "missed_detection": 1.0,
                "unnecessary_wait": 1.0,
                "observe_cost": 1.0,
            }
        fp_rate = fb["false_positives"] / fb["total"]
        fn_rate = fb["false_negatives"] / fb["total"]
        return {
            "false_positive_act": 1.0 + fp_rate,
            "missed_detection": 1.0 + fn_rate,
            "unnecessary_wait": 1.0,
            "observe_cost": 1.0,
        }

    # ---------- weight resolution ----------

    def _get_loss_weights(
        self,
        crop: str,
        growth_stage: Optional[str] = None,
        season: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get dynamically-adjusted crop-specific loss weights."""
        if self._custom_weights:
            base = dict(self._custom_weights)
        else:
            base = dict(
                DEFAULT_LOSS_WEIGHTS.get(
                    crop.lower(),
                    DEFAULT_LOSS_WEIGHTS["default"],
                )
            )

        # Apply growth-stage multiplier
        gs_key = (growth_stage or "").lower()
        gs_mult = GROWTH_STAGE_MULTIPLIERS.get(
            gs_key,
            GROWTH_STAGE_MULTIPLIERS.get("vegetative", {}),
        )
        for k in base:
            base[k] *= gs_mult.get(k, 1.0)

        # Apply season multiplier
        s_key = (season or "").lower()
        s_mult = SEASON_MULTIPLIERS.get(
            s_key,
            SEASON_MULTIPLIERS["default"],
        )
        for k in base:
            base[k] *= s_mult.get(k, 1.0)

        # Apply feedback multiplier
        fb_mult = self._feedback_multipliers(crop)
        for k in base:
            base[k] *= fb_mult.get(k, 1.0)

        return base

    # ----- Confidence Band (bootstrap percentile) -----

    def compute_confidence_band(
        self,
        signals: List[float],
        n_bootstrap: int = 1000,
        ci_lower: float = 0.05,
        ci_upper: float = 0.95,
    ) -> ConfidenceBand:
        """
        Compute bootstrap percentile confidence band from multiple signal
        confidence scores.

        Parameters
        ----------
        signals : list of confidence scores from different pipeline components
        n_bootstrap : number of bootstrap resamples
        ci_lower : lower percentile (default 5%)
        ci_upper : upper percentile (default 95%)

        Returns
        -------
        ConfidenceBand with lower, median, upper
        """
        if not signals:
            return ConfidenceBand(lower=0.0, median=0.0, upper=0.0)

        arr = np.array(signals, dtype=np.float64)
        rng = np.random.default_rng(42)

        bootstrap_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)])

        lower = float(np.percentile(bootstrap_means, ci_lower * 100))
        median = float(np.percentile(bootstrap_means, 50))
        upper = float(np.percentile(bootstrap_means, ci_upper * 100))

        return ConfidenceBand(
            lower=max(0.0, min(1.0, lower)),
            median=max(0.0, min(1.0, median)),
            upper=max(0.0, min(1.0, upper)),
        )

    # ----- Regret Scoring -----

    def compute_regret(
        self,
        action: DecisionAction,
        confidence: float,
        crop: str,
        has_anomaly: bool = False,
        growth_stage: Optional[str] = None,
        season: Optional[str] = None,
    ) -> float:
        """
        Compute minimax regret for a candidate action.

        regret = max(0, expected_loss(action) - expected_loss(best_alternative))

        Parameters
        ----------
        action : candidate action
        confidence : pipeline confidence score
        crop : crop type for loss weights
        has_anomaly : whether anomaly gate flagged

        Returns
        -------
        float: regret score (0 = no regret, higher = worse)
        """
        weights = self._get_loss_weights(
            crop,
            growth_stage,
            season,
        )  # dynamic

        # Expected loss for each action given current confidence
        losses: Dict[DecisionAction, float] = {
            DecisionAction.ACT: (
                weights["false_positive_act"] * (1 - confidence)
                + (weights["missed_detection"] * 0.1 if has_anomaly else 0)
            ),
            DecisionAction.WAIT: (
                weights["unnecessary_wait"] * confidence + weights["missed_detection"] * (1 - confidence) * 0.3
            ),
            DecisionAction.OBSERVE: (weights["observe_cost"] + weights["missed_detection"] * (1 - confidence) * 0.5),
            DecisionAction.DO_NOTHING: (weights["missed_detection"] * confidence),
        }

        # Penalize ACT heavily if anomaly detected
        if has_anomaly:
            losses[DecisionAction.ACT] += 1.0

        best_loss = min(losses.values())
        return max(0.0, losses[action] - best_loss)

    # ----- Main Decision Method -----

    def decide(self, inputs: Dict[str, Any]) -> GovernorDecision:
        """
        Make a decision given pipeline inputs.

        Parameters
        ----------
        inputs : dict with keys:
            - vision_result: VisionResult or None
            - sensor_data: dict or None
            - vrag_evidence: List[VRAGResult] or None
            - anomaly_flag: AnomalyFlag or None
            - crop_context: dict with 'crop_type', 'disease' etc.

        Returns
        -------
        GovernorDecision
        """
        vision: Optional[VisionResult] = inputs.get("vision_result")
        sensor_data: Optional[Dict] = inputs.get("sensor_data")
        vrag_evidence: List[VRAGResult] = inputs.get("vrag_evidence") or []
        anomaly: Optional[AnomalyFlag] = inputs.get("anomaly_flag")
        crop_ctx: Dict[str, Any] = inputs.get("crop_context") or {}

        crop = crop_ctx.get("crop_type", "default")
        disease = crop_ctx.get("disease", "unknown")
        growth_stage = crop_ctx.get("growth_stage")
        season = crop_ctx.get("season")

        # Collect confidence signals
        signals: List[float] = []
        evidence: List[str] = []

        if vision:
            signals.append(vision.confidence)
            evidence.append(f"Vision confidence: {vision.confidence:.2f}")
            if vision.disease_name:
                evidence.append(f"Detected: {vision.disease_name}")

        if sensor_data:
            # Normalize sensor readings to [0, 1] confidence proxy
            sensor_conf = self._sensor_confidence(sensor_data)
            signals.append(sensor_conf)
            evidence.append(f"Sensor data confidence: {sensor_conf:.2f}")

        if vrag_evidence:
            avg_sim = np.mean([r.similarity_score for r in vrag_evidence])
            signals.append(float(avg_sim))
            evidence.append(f"VRAG evidence: {len(vrag_evidence)} matches (avg similarity: {avg_sim:.3f})")
            for r in vrag_evidence[:3]:
                evidence.append(f"  → {r.evidence_text}")

        has_anomaly = False
        if anomaly:
            has_anomaly = anomaly.is_anomaly
            if has_anomaly:
                evidence.append(f"⚠️ ANOMALY: {anomaly.reason}")
            else:
                evidence.append("✓ Input within known distribution")

        # Compute confidence band
        band = self.compute_confidence_band(signals if signals else [0.0])
        evidence.append(f"Confidence band: [{band.lower:.2f}, {band.median:.2f}, {band.upper:.2f}]")

        # Decision logic
        if has_anomaly:
            # Anomaly gate: cap at OBSERVE
            action = DecisionAction.OBSERVE
            evidence.append("Action capped at OBSERVE due to anomaly gate")
        elif band.lower < 0.4:
            action = DecisionAction.DO_NOTHING
            evidence.append("Low confidence — DO_NOTHING")
        elif band.lower < 0.6:
            action = DecisionAction.OBSERVE
            evidence.append("Moderate confidence — OBSERVE")
        elif band.lower < 0.8:
            action = DecisionAction.WAIT
            evidence.append("Good confidence but not decisive — WAIT for more data")
        else:
            action = DecisionAction.ACT
            evidence.append("High confidence — ACT recommended")

        # Compute regret for chosen action and alternatives
        regret = self.compute_regret(
            action,
            band.median,
            crop,
            has_anomaly,
            growth_stage,
            season,
        )

        alternatives = []
        for alt_action in DecisionAction:
            if alt_action == action:
                continue
            alt_regret = self.compute_regret(
                alt_action,
                band.median,
                crop,
                has_anomaly,
                growth_stage,
                season,
            )
            alternatives.append(
                {
                    "action": alt_action.value,
                    "regret_score": round(alt_regret, 4),
                }
            )
        alternatives.sort(key=lambda x: x["regret_score"])

        # Get treatment template if ACT
        treatment = None
        if action == DecisionAction.ACT and disease != "unknown":
            template = get_action_template(crop, disease)
            if template:
                treatment = template.model_dump()

        return GovernorDecision(
            action=action,
            confidence_band=band,
            regret_score=round(regret, 4),
            evidence=evidence,
            alternative_actions=alternatives,
            treatment=treatment,
        )

    def _sensor_confidence(self, sensor_data: Dict[str, Any]) -> float:
        """
        Convert sensor readings to a confidence proxy.
        Checks if values are within normal agricultural ranges.
        """
        checks_passed = 0
        total_checks = 0

        ranges = {
            "temperature": (10, 40),
            "humidity": (20, 90),
            "soil_moisture": (20, 80),
            "ph": (5.5, 8.0),
            "nitrogen": (0, 200),
            "phosphorus": (0, 100),
            "potassium": (0, 200),
        }

        for key, (low, high) in ranges.items():
            if key in sensor_data:
                total_checks += 1
                val = float(sensor_data[key])
                if low <= val <= high:
                    checks_passed += 1

        if total_checks == 0:
            return 0.5  # no data → neutral
        return checks_passed / total_checks
