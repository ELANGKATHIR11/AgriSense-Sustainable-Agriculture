# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps MLOps Service Layer
Manages experiments, challenger validation, model promotion, and GPU deployment configurations.
"""

import logging
from typing import Dict, Any
from sqlalchemy.orm import Session
from backend.database.models import ModelRegistry, PredictionLog
from backend.agriops.common.event_bus import event_bus
from backend.agriops.telemetry.tracer import trace_span

logger = logging.getLogger("AgriOps.MLOps")


class MLOpsService:
    @trace_span("MLOps.EvaluateChallenger")
    async def evaluate_champion_challenger(
        self, db: Session, model_type: str
    ) -> Dict[str, Any]:
        """
        Retrieves active champion vs staging challenger models to evaluate metrics.
        """
        champion = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.type == model_type, ModelRegistry.status == "active")
            .first()
        )

        challenger = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.type == model_type, ModelRegistry.status == "staging")
            .first()
        )

        if not champion:
            return {
                "status": "error",
                "message": "No active champion model found for type: " + model_type,
            }

        comparison = {
            "model_type": model_type,
            "champion": {
                "id": champion.id,
                "name": champion.name,
                "version": champion.version,
                "accuracy": champion.accuracy,
                "f1_score": champion.f1_score,
                "prediction_count": champion.prediction_count,
            },
            "challenger": None,
            "promotion_recommended": False,
        }

        if challenger:
            # Recommend promotion if challenger accuracy is higher
            promo = challenger.accuracy > champion.accuracy
            comparison["challenger"] = {
                "id": challenger.id,
                "name": challenger.name,
                "version": challenger.version,
                "accuracy": challenger.accuracy,
                "f1_score": challenger.f1_score,
            }
            comparison["promotion_recommended"] = promo

        return comparison

    @trace_span("MLOps.PromoteChallenger")
    async def promote_model(self, db: Session, challenger_id: str) -> Dict[str, Any]:
        """
        Promotes a challenger model to active, archiving the old champion model.
        """
        challenger = (
            db.query(ModelRegistry).filter(ModelRegistry.id == challenger_id).first()
        )
        if not challenger:
            return {"status": "error", "message": "Challenger model not found"}

        # Find existing active model of the same type
        champion = (
            db.query(ModelRegistry)
            .filter(
                ModelRegistry.type == challenger.type, ModelRegistry.status == "active"
            )
            .first()
        )

        if champion:
            champion.status = "archived"

        challenger.status = "active"
        db.commit()

        await event_bus.publish(
            "ModelPromoted",
            {
                "model_id": challenger.id,
                "model_name": challenger.name,
                "model_type": challenger.type,
                "version": challenger.version,
            },
        )

        return {
            "status": "success",
            "promoted_model_id": challenger.id,
            "active_version": challenger.version,
            "previous_champion_archived": champion.id if champion else None,
        }

    @trace_span("MLOps.GetPerformanceMetrics")
    def get_performance_analytics(self, db: Session) -> Dict[str, Any]:
        """
        Aggregates inference volume, latency trends, and drift scores from PredictionLog.
        """
        logs = (
            db.query(PredictionLog)
            .order_by(PredictionLog.timestamp.desc())
            .limit(100)
            .all()
        )

        if not logs:
            return {
                "inference_count": 0,
                "avg_latency_ms": 0.0,
                "avg_confidence": 0.0,
                "avg_drift_score": 0.0,
            }

        latencies = [log.latency_ms for log in logs if log.latency_ms is not None]
        confidences = [log.confidence for log in logs if log.confidence is not None]
        drifts = [log.drift_score for log in logs if log.drift_score is not None]

        return {
            "inference_count": len(logs),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2)
            if latencies
            else 0.0,
            "avg_confidence": round(sum(confidences) / len(confidences), 3)
            if confidences
            else 0.0,
            "avg_drift_score": round(sum(drifts) / len(drifts), 3) if drifts else 0.0,
        }


mlops_service = MLOpsService()
