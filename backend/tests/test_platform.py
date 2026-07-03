"""
AGRISENSE Automated Platform Test Suite
Verifies DB operations, ML inference, RAG engine retrieval, EIF anomalies, and Twin pipeline.
Run: pytest backend/tests/
"""

import os
import sys

# Mock sentence_transformers and faiss to force lightweight offline fallbacks in RAG engine
sys.modules["sentence_transformers"] = None
sys.modules["faiss"] = None

import numpy as np
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.main import app
from backend.database import get_db
from ml.extended_isolation_forest import ExtendedIsolationForest
from ml.inference import (
    predict_crop,
    predict_fertilizer,
    predict_yield,
    predict_irrigation,
    predict_disease_risk,
)
from backend.rag.rag_engine import rag_engine

client = TestClient(app)


def test_database_connection():
    """Verify database connection and schema initialization."""
    db = next(get_db())
    assert db is not None
    db.close()


def test_ml_inference():
    """Verify prediction outputs for all tabular models."""
    # 1. Crop recommendation
    crop_res = predict_crop(
        N=50, P=40, K=40, temperature=28.0, humidity=60.0, ph=6.5, rainfall=100.0
    )
    assert "crops" in crop_res
    assert len(crop_res["crops"]) > 0
    assert "name" in crop_res["crops"][0]

    # 2. Fertilizer recommendation
    fert_res = predict_fertilizer(
        temperature=28.0,
        humidity=60.0,
        moisture=38.0,
        soil_type="Loamy",
        crop_type="Rice",
        nitrogen=50,
        potassium=40,
        phosphorus=40,
    )
    assert "recommendedFertilizer" in fert_res
    assert "confidence" in fert_res

    # 3. Yield prediction
    yield_res = predict_yield(
        area_acres=10.0,
        avg_rainfall=1000.0,
        avg_temp=28.0,
        crop_type="Rice",
        nitrogen=50,
        phosphorus=40,
        potassium=40,
    )
    assert "predictedYieldTons" in yield_res
    assert yield_res["predictedYieldTons"] > 0

    # 4. Irrigation prediction
    irr_res = predict_irrigation(
        moisture=38.0, temperature=28.0, humidity=60.0, crop_type="Rice"
    )
    assert "waterRequiredLiters" in irr_res

    # 5. Disease risk prediction
    disease_res = predict_disease_risk(
        humidity=60.0,
        temperature=28.0,
        leaf_wetness_hours=8.0,
        soil_moisture=38.0,
        rainfall_mm=5.0,
    )
    assert "riskScore" in disease_res


def test_extended_isolation_forest():
    """Verify EIF anomaly scores detect out-of-bounds inputs."""
    # Create healthy baseline: 100 normal vectors
    np.random.seed(42)
    healthy = np.random.normal(loc=10.0, scale=1.0, size=(100, 3))

    eif = ExtendedIsolationForest(n_estimators=10, max_samples=64)
    eif.fit(healthy)

    # Normal sample should have low anomaly score
    normal_sample = np.array([[10.1, 9.9, 10.0]])
    score_normal = eif.compute_anomaly_score(normal_sample)[0]

    # Outlier sample should have higher anomaly score
    outlier_sample = np.array([[25.0, -10.0, 50.0]])
    score_outlier = eif.compute_anomaly_score(outlier_sample)[0]

    assert score_outlier > score_normal


def test_rag_engine():
    """Verify RAG engine query returns relevant records."""
    import asyncio

    results = asyncio.run(
        rag_engine.query_disease_kb("Tomato Leaf Mold symptoms", top_k=1)
    )
    assert len(results) > 0
    assert "disease_name" in results[0]


def test_digital_twin_pipeline():
    """Verify digital twin telemetry execution pipeline."""
    telemetry = {
        "nitrogen": 45,
        "phosphorus": 38,
        "potassium": 42,
        "temperature": 28.5,
        "humidity": 62.0,
        "pH": 6.4,
        "soilMoisture": 38.3,
        "rainfall": 0.0,
        "windSpeed": 8.4,
    }
    # Test POST endpoint
    response = client.post("/api/twin/update", json=telemetry)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "updated"
    assert "anomalyScore" in data
    assert "twinState" in data
    assert "confidenceInterval" in data["twinState"]["waterTwin"]
    assert "recommendationNotes" in data["twinState"]
