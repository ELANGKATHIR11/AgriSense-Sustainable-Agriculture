"""
AGRISENSE Master Model Training Pipeline
Trains ALL models (CatBoost and PyTorch PatchTST) using cleaned datasets and saves accuracy metrics.
Run from project root: python ml/train_all_models.py
"""

import os
import sys
import json
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

METRICS_PATH = os.path.join(os.path.dirname(__file__), "models", "metrics.json")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

metrics = {}

def save_metrics():
    metrics["last_trained"] = datetime.utcnow().isoformat() + "Z"
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

# ── 1. CROP RECOMMENDATION ──────────────────────────────────────────────────
print("\n" + "="*60)
print("  [1/5] CROP RECOMMENDATION — CatBoost Classifier")
print("="*60)
t0 = time.time()

from ml.crop_training import train_crop_model
crop_acc, crop_f1 = train_crop_model()
metrics["crop_recommendation"] = {
    "model": "CatBoostClassifier",
    "accuracy": round(crop_acc, 4),
    "f1_score": round(crop_f1, 4),
    "train_time_s": round(time.time() - t0, 1),
    "status": "active" if crop_acc >= 0.80 else "below_target"
}

# ── 2. FERTILIZER RECOMMENDATION ─────────────────────────────────────────────
print("\n" + "="*60)
print("  [2/5] FERTILIZER RECOMMENDATION — CatBoost Classifier")
print("="*60)
t0 = time.time()

from ml.fertilizer_training import train_fertilizer_model
fert_acc, fert_f1 = train_fertilizer_model()
metrics["fertilizer_recommendation"] = {
    "model": "CatBoostClassifier",
    "accuracy": round(fert_acc, 4),
    "f1_score": round(fert_f1, 4),
    "train_time_s": round(time.time() - t0, 1),
    "status": "active" if fert_acc >= 0.80 else "below_target"
}

# ── 3. IRRIGATION OPTIMIZATION ─────────────────────────────────────────────
print("\n" + "="*60)
print("  [3/5] IRRIGATION — PatchTST PyTorch Sequence Transformer")
print("="*60)
t0 = time.time()

from ml.irrigation_training import train_irrigation_model
irr_r2, irr_mae = train_irrigation_model()
metrics["irrigation"] = {
    "model": "PatchTST Sequence Transformer",
    "r2_score": round(irr_r2, 4),
    "mae_liters": round(irr_mae, 2),
    "train_time_s": round(time.time() - t0, 1),
    "status": "active" if irr_r2 >= 0.70 else "below_target"
}

# ── 4. YIELD PREDICTION ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("  [4/5] YIELD PREDICTION — PatchTST PyTorch Sequence Transformer")
print("="*60)
t0 = time.time()

from ml.yield_training import train_yield_model
yld_r2, yld_mae = train_yield_model()
metrics["yield_prediction"] = {
    "model": "PatchTST Sequence Transformer",
    "r2_score": round(yld_r2, 4),
    "mae_tons": round(yld_mae, 4),
    "train_time_s": round(time.time() - t0, 1),
    "status": "active" if yld_r2 >= 0.70 else "below_target"
}

# ── 5. DISEASE RISK ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  [5/5] DISEASE RISK — Tabular Classifier")
print("="*60)
t0 = time.time()

from ml.disease_risk_training import train_disease_risk_model
dis_acc, dis_auc = train_disease_risk_model()
metrics["disease_risk"] = {
    "model": "RandomForest + LightGBM (ROC-AUC optimised)",
    "accuracy": round(dis_acc, 4),
    "roc_auc": round(dis_auc, 4),
    "train_time_s": round(time.time() - t0, 1),
    "status": "active" if dis_acc >= 0.80 else "below_target"
}

# ── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TRAINING COMPLETE — SUMMARY")
print("="*60)
for name, m in metrics.items():
    if name == "last_trained":
        continue
    status_icon = "✅" if m.get("status") == "active" else "⚠️"
    if "accuracy" in m:
        print(f"  {status_icon} {name}: {m['accuracy']*100:.2f}% accuracy ({m['model']})")
    else:
        print(f"  {status_icon} {name}: R²={m['r2_score']:.4f} ({m['model']})")

save_metrics()

if __name__ == "__main__":
    pass
