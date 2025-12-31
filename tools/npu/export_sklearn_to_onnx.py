"""
Export sklearn joblib models to ONNX and convert to OpenVINO IR (if OpenVINO available).

Usage: python tools/npu/export_sklearn_to_onnx.py

Finds models in `agrisense_app/backend/models/` matching `*_rf_npu.joblib` and `*_gb_npu.joblib` and
attempts to export them to ONNX using `skl2onnx`. Saves ONNX and then converts to OpenVINO IR under
`agrisense_app/backend/models/openvino_npu/`.
"""
import os
import sys
from pathlib import Path
import joblib
import numpy as np

MODELS_DIR = Path("agrisense_app/backend/models")
OUT_DIR = MODELS_DIR / "openvino_npu"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def try_convert(model_path: Path, n_features: int = 7):
    name = model_path.stem
    print(f"\n-- Converting {name}")
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as e:
        print("skl2onnx not available:", e)
        return False

    model = joblib.load(model_path)

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_path = OUT_DIR / f"{name}.onnx"
    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"✅ ONNX saved: {onnx_path}")
    except Exception as e:
        print(f"⚠️ ONNX conversion failed for {name}: {e}")
        return False

    # Try to convert ONNX to OpenVINO IR if openvino is installed
    try:
        import openvino as ov
        try:
            # modern API
            from openvino import convert_model
            ov_model = convert_model(onnx_path)
            save_dir = OUT_DIR / name
            save_dir.mkdir(exist_ok=True)
            ov.save_model(ov_model, str(save_dir / f"{name}.xml"))
            print(f"✅ OpenVINO IR saved: {save_dir}")
        except Exception:
            # fallback to legacy mo
            from openvino.tools import mo
            ov_model = mo.convert_model(onnx_path)
            save_dir = OUT_DIR / name
            save_dir.mkdir(exist_ok=True)
            import openvino as _ov
            _ov.save_model(ov_model, str(save_dir / f"{name}.xml"))
            print(f"✅ OpenVINO IR saved (legacy): {save_dir}")
    except Exception as e:
        print(f"⚠️ OpenVINO conversion skipped or failed: {e}")

    return True


def main():
    # feature count heuristics: try scaler to detect
    scaler_path = MODELS_DIR / "crop_scaler.joblib"
    n_features = 7
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            if hasattr(scaler, "scale_") and hasattr(scaler, "n_features_in_"):
                n_features = int(getattr(scaler, "n_features_in_", n_features))
        except Exception:
            pass

    candidates = list(MODELS_DIR.glob("*_rf_npu.joblib")) + list(MODELS_DIR.glob("*_gb_npu.joblib"))
    if not candidates:
        print("No sklearn models found to convert in", MODELS_DIR)
        return

    for m in candidates:
        try_convert(m, n_features=n_features)


if __name__ == "__main__":
    main()
