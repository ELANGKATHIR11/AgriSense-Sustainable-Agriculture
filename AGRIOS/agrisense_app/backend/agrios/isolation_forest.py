"""
AGRI-OS Isolation Forest — Anomaly Detection & OOD Gating
==========================================================
Trains an Isolation Forest on known-good embeddings and gates the
Decision Governor to OBSERVE when out-of-distribution inputs are detected.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .schemas import AnomalyFlag, DecisionAction

logger = logging.getLogger("agrios.anomaly")


class AnomalyGate:
    """
    Isolation Forest-based anomaly detector for DeiT embeddings.

    Trains on known-good embeddings (from disease/crop datasets).
    At inference: if an embedding is OOD, the Decision Governor is forced
    to OBSERVE — it can never ACT on anomalous inputs.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies (default 0.05 = 5%)
    n_estimators : int
        Number of isolation trees (default 100)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        from sklearn.ensemble import IsolationForest

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._trained = False

    def train(
        self,
        embeddings: np.ndarray,
        save_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Train the Isolation Forest on known-good embeddings.

        Parameters
        ----------
        embeddings : np.ndarray (N, 384)
            Known-good disease/crop embeddings from the training set.
        save_path : optional
            Path to save the trained model (.joblib).

        Returns
        -------
        dict with training stats
        """
        n_samples, n_features = embeddings.shape
        logger.info(
            "Training Isolation Forest on %d samples (%d features)",
            n_samples,
            n_features,
        )

        self.model.fit(embeddings)
        self._trained = True

        # Compute stats on training data
        scores = self.model.decision_function(embeddings)
        predictions = self.model.predict(embeddings)
        n_anomalies = int((predictions == -1).sum())

        stats = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_anomalies_in_training": n_anomalies,
            "anomaly_rate": n_anomalies / n_samples if n_samples > 0 else 0.0,
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }

        if save_path:
            import joblib

            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, str(save_path))
            logger.info("Anomaly gate saved to %s", save_path)
            stats["model_path"] = str(save_path)

        logger.info("Anomaly gate trained: %s", stats)
        return stats

    def load(self, model_path: str | Path) -> None:
        """Load a previously trained Isolation Forest."""
        import joblib

        model_path = str(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Anomaly gate model not found: {model_path}")

        self.model = joblib.load(model_path)
        self._trained = True
        logger.info("Anomaly gate loaded from %s", model_path)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def gate(self, embedding: np.ndarray) -> AnomalyFlag:
        """
        Evaluate a single embedding through the anomaly gate.

        Parameters
        ----------
        embedding : np.ndarray (384,)
            DeiT embedding to evaluate.

        Returns
        -------
        AnomalyFlag with is_anomaly, score, and recommended gate action.
        """
        if not self._trained:
            # Not trained → pass through (no gating)
            return AnomalyFlag(
                is_anomaly=False,
                anomaly_score=0.0,
                gate_action=DecisionAction.ACT,
                reason="Anomaly gate not trained — pass-through mode",
            )

        x = embedding.reshape(1, -1).astype(np.float32)

        # decision_function: negative = anomaly, positive = normal
        score = float(self.model.decision_function(x)[0])
        prediction = int(self.model.predict(x)[0])
        is_anomaly = prediction == -1

        if is_anomaly:
            gate_action = DecisionAction.OBSERVE
            reason = (
                f"OOD input detected (score={score:.4f}). "
                "Decision Governor capped at OBSERVE — cannot ACT on anomalous inputs."
            )
        else:
            gate_action = DecisionAction.ACT  # gate open, Governor decides freely
            reason = f"Input within known distribution (score={score:.4f})"

        return AnomalyFlag(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            gate_action=gate_action,
            reason=reason,
        )

    def batch_gate(self, embeddings: np.ndarray) -> list[AnomalyFlag]:
        """Evaluate multiple embeddings."""
        results = []
        for emb in embeddings:
            results.append(self.gate(emb))
        return results
