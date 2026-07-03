# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field
from ml.extended_isolation_forest import ExtendedIsolationForest

router = APIRouter(prefix="/ml/anomaly", tags=["Anomaly Detection"])


# Standard input for single checking
class AnomalyCheckInput(BaseModel):
    soil_moisture: float = Field(..., ge=0.0, le=100.0)
    temperature: float = Field(..., ge=-20.0, le=60.0)
    humidity: float = Field(..., ge=0.0, le=100.0)
    pH: float = Field(..., ge=0.0, le=14.0)
    nitrogen: float = Field(..., ge=0)
    phosphorus: float = Field(..., ge=0)
    potassium: float = Field(..., ge=0)


_eif_model = None
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ml",
    "models",
    "eif_detector.joblib",
)


def load_or_train_eif():
    global _eif_model
    if _eif_model is not None:
        return _eif_model

    if os.path.exists(MODEL_PATH):
        try:
            _eif_model = joblib.load(MODEL_PATH)
            return _eif_model
        except Exception:
            pass

    # Fit a baseline with mock/seeded normal telemetry distribution
    np.random.seed(42)
    # Generate 300 normal sensor readings
    moisture = np.random.normal(40, 5, 300)
    temp = np.random.normal(28, 3, 300)
    humidity = np.random.normal(60, 8, 300)
    ph = np.random.normal(6.5, 0.4, 300)
    n = np.random.normal(45, 5, 300)
    p = np.random.normal(38, 4, 300)
    k = np.random.normal(42, 5, 300)

    X_train = np.column_stack([moisture, temp, humidity, ph, n, p, k])
    _eif_model = ExtendedIsolationForest(n_estimators=100, max_samples=256)
    _eif_model.fit(X_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(_eif_model, MODEL_PATH)
    return _eif_model


@router.post("/check")
async def check_anomaly(payload: AnomalyCheckInput):
    model = load_or_train_eif()
    x = np.array(
        [
            payload.soil_moisture,
            payload.temperature,
            payload.humidity,
            payload.pH,
            payload.nitrogen,
            payload.phosphorus,
            payload.potassium,
        ]
    ).reshape(1, -1)

    score = model.compute_anomaly_score(x)[0]
    # In EIF, score close to 1 means highly anomalous, score below 0.5 is normal
    is_anomaly = bool(score > 0.62)

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": float(score),
        "status": "ANOMALOUS" if is_anomaly else "NORMAL",
        "threshold": 0.62,
    }
