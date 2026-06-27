import os
import sys
import types
from unittest.mock import MagicMock

# ── Add project root to sys.path ───────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ── Mock Torch & PyTorch nn.Module for subclassing safety ──────────────────
class MockModule:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def forward(self, *args, **kwargs):
        return args[0] if args else None

mock_nn = MagicMock()
mock_nn.Module = MockModule
sys.modules['torch.nn'] = mock_nn

mock_torch = MagicMock()
mock_torch.nn = mock_nn
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available.return_value = False
sys.modules['torch'] = mock_torch

# ── Mock pandas & PIL ──────────────────────────────────────────────────────
sys.modules['pandas'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()

# ── Mock ML libraries ──────────────────────────────────────────────────────
sys.modules['joblib'] = MagicMock()
sys.modules['xgboost'] = MagicMock()
sys.modules['lightgbm'] = MagicMock()
sys.modules['catboost'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# Disable sentence_transformers and faiss to force offline TF-IDF/fallback paths
sys.modules['sentence_transformers'] = None
sys.modules['faiss'] = None

# ── Mock ml.inference package completely to bypass joblib loading ─────────
ml_module = types.ModuleType("ml")
inference_module = types.ModuleType("ml.inference")

inference_module.predict_crop = lambda *a, **k: {"crops": [{"name": "Rice", "suitability": 95.0, "description": "Mocked Crop"}]}
inference_module.predict_fertilizer = lambda *a, **k: {"recommendedFertilizer": "Urea", "confidence": 92.5}
inference_module.predict_yield = lambda *a, **k: {"predictedYieldTons": 4.5}
inference_module.predict_irrigation = lambda *a, **k: {"waterRequiredLiters": 1200.0}
inference_module.predict_disease_risk = lambda *a, **k: {"riskScore": 15.0}

ml_module.inference = inference_module
sys.modules["ml"] = ml_module
sys.modules["ml.inference"] = inference_module
