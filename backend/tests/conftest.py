# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

import os
import sys
import types
from unittest.mock import MagicMock

# Mock google.antigravity
mock_antigravity = MagicMock()
mock_antigravity.Agent = MagicMock()
mock_antigravity.LocalAgentConfig = MagicMock()
mock_antigravity.CapabilitiesConfig = MagicMock()
mock_antigravity.ModelTarget = MagicMock()
mock_antigravity.ModelEndpoint = MagicMock()
sys.modules["google.antigravity"] = mock_antigravity

mock_conv = MagicMock()
mock_conv.Conversation = MagicMock()
sys.modules["google.antigravity.conversation"] = mock_conv
sys.modules["google.antigravity.conversation.conversation"] = mock_conv


# ── Add project root to sys.path ────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

# ── Set DATABASE_URL to PostgreSQL for all test sessions ────────────────────
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://postgres:Akilaarasu1!@127.0.0.1:5432/agriops"
)
os.environ.setdefault(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres:Akilaarasu1!@127.0.0.1:5432/agriops",
)

# ── Real pandas, lancedb, pyarrow needed — do NOT mock ──────────────────────
# Only mock PIL since image display is not needed in tests
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()

# ── Mock heavy vision libs not needed for API/integration tests ──────────────
sys.modules["cv2"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()


# ── Mock torch to skip GPU init during tests ────────────────────────────────
class MockTorchModule:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def forward(self, *args, **kwargs):
        return args[0] if args else None


mock_nn = MagicMock()
mock_nn.Module = MockTorchModule
sys.modules["torch.nn"] = mock_nn

mock_torch = MagicMock()
mock_torch.nn = mock_nn
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available.return_value = True
mock_torch.__version__ = "2.11.0+cu128"

class MockTensor:
    pass

mock_torch.Tensor = MockTensor
sys.modules["torch"] = mock_torch

# ── Mock onnxruntime (GPU version) ───────────────────────────────────────────
mock_ort = MagicMock()
mock_ort.InferenceSession = MagicMock(
    return_value=MagicMock(
        run=MagicMock(return_value=[[[0.1, 0.9]]]),
        get_inputs=MagicMock(return_value=[MagicMock(name="input")]),
        get_outputs=MagicMock(return_value=[MagicMock(name="output")]),
    )
)
sys.modules["onnxruntime"] = mock_ort

# ── Mock joblib/catboost/xgboost/lightgbm (not needed for API tests) ────────
sys.modules["joblib"] = MagicMock()
sys.modules["xgboost"] = MagicMock()
sys.modules["lightgbm"] = MagicMock()
sys.modules["catboost"] = MagicMock()

# ── sentence_transformers and faiss: force offline TF-IDF paths ──────────────
sys.modules["sentence_transformers"] = None
sys.modules["faiss"] = None

# ── ml package: proper package-style mock ────────────────────────────────────
ml_module = types.ModuleType("ml")
ml_module.__path__ = []
ml_module.__package__ = "ml"

# ml.inference submodule
inference_module = types.ModuleType("ml.inference")
inference_module.__package__ = "ml"
inference_module.predict_crop = lambda *a, **k: {
    "crops": [{"name": "Rice", "suitability": 95.0, "description": "Mocked Crop"}]
}
inference_module.predict_fertilizer = lambda *a, **k: {
    "recommendedFertilizer": "Urea",
    "confidence": 92.5,
}
inference_module.predict_yield = lambda *a, **k: {"predictedYieldTons": 4.5}
inference_module.predict_irrigation = lambda *a, **k: {"waterRequiredLiters": 1200.0}
inference_module.predict_disease_risk = lambda *a, **k: {"riskScore": 15.0}

# ml.extended_isolation_forest submodule
eif_module = types.ModuleType("ml.extended_isolation_forest")
eif_module.__package__ = "ml"


class ExtendedIsolationForest:
    """Mock ExtendedIsolationForest satisfying the AgriSense EIF API."""

    def __init__(self, *args, **kwargs):
        self.n_estimators = kwargs.get("n_estimators", 100)
        self._fitted = False

    def fit(self, X, *args, **kwargs):
        self._fitted = True
        return self

    def predict(self, X, *args, **kwargs):
        try:
            n = len(X)
        except Exception:
            n = 1
        return [1] * n  # 1 = inlier

    def score_samples(self, X, *args, **kwargs):
        try:
            n = len(X)
        except Exception:
            n = 1
        return [-0.5] * n

    def compute_anomaly_score(self, X, *args, **kwargs):
        """
        Returns anomaly scores based on distance from normal range.
        Normal (mean ~10) -> low score ~0.3
        Outlier (extreme values) -> high score ~0.8
        """
        try:
            import numpy as np

            X_arr = np.array(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            center = 10.0
            scales = np.abs(X_arr - center).mean(axis=1)
            scores = np.clip(0.2 + (scales / 30.0), 0.2, 0.9)
            return scores
        except Exception:
            return [0.3]

    def decision_function(self, X, *args, **kwargs):
        try:
            n = len(X)
        except Exception:
            n = 1
        return [0.0] * n


eif_module.ExtendedIsolationForest = ExtendedIsolationForest

# ml.models (catch-all)
models_module = types.ModuleType("ml.models")
models_module.__package__ = "ml"

# Wire onto package
ml_module.inference = inference_module
ml_module.extended_isolation_forest = eif_module
ml_module.models = models_module

sys.modules["ml"] = ml_module
sys.modules["ml.inference"] = inference_module
sys.modules["ml.extended_isolation_forest"] = eif_module
sys.modules["ml.models"] = models_module

# ── pytest fixtures ──────────────────────────────────────────────────────────
import pytest

# PostgreSQL connection — single source of truth
PG_URL = "postgresql+psycopg://postgres:Akilaarasu1!@127.0.0.1:5432/agriops"


@pytest.fixture(scope="session")
def db():
    """
    Provide a PostgreSQL SQLAlchemy session for tests.
    All tables already exist in PostgreSQL (including PostGIS geography tables).
    """
    from backend.database.session import SessionLocalSync

    session = SessionLocalSync()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
