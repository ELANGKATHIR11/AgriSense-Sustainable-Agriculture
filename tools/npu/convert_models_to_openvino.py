"""
Convert trained models to OpenVINO IR.

- Converts scikit-learn joblib models to ONNX using `skl2onnx` (if available) and then to OpenVINO IR.
- Converts PyTorch `.pt` models to ONNX then to OpenVINO IR (if `torch` and `openvino` available).

If OpenVINO/ONNX tools are missing, the script prints recommended commands.
"""
import os
import sys
from pathlib import Path
import subprocess

MODELS_DIR = Path("agrisense_app/backend/models")
OUT_DIR = Path("models/openvino_npu")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def infer_feature_count(models_dir: Path) -> int:
    # Try several heuristics to determine input feature count for tabular models
    # 1) look for scaler.pkl or label_encoder.pkl with attribute n_features_in_
    # 2) read a common csv dataset in backend/data
    # 3) fallback to 8
    try:
        import joblib
        scaler_path = models_dir.parent.parent / 'data' / 'scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            if hasattr(scaler, 'n_features_in_'):
                return int(scaler.n_features_in_)
    except Exception:
        pass
    # heuristic: try crop dataset
    csv_path = models_dir.parent.parent / 'data' / 'Crop_recommendation.csv'
    try:
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                hdr = f.readline().strip().split(',')
                # exclude label column if present
                if 'label' in [c.lower() for c in hdr]:
                    return max(1, len(hdr) - 1)
                return max(1, len(hdr))
    except Exception:
        pass
    return 8


def run(cmd):
    print(f"> {cmd}")
    return subprocess.run(cmd, shell=True)


def convert_sklearn(model_path, name):
    print(f"\n-- Converting sklearn model: {model_path}")
    onnx_path = OUT_DIR / f"{name}.onnx"
    xml_path = OUT_DIR / f"{name}.xml"
    # Attempt skl2onnx
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import joblib
        print("skl2onnx available — exporting to ONNX")
        model = joblib.load(model_path)
        # infer number of features
        n_features = None
        if hasattr(model, 'n_features_in_'):
            n_features = int(model.n_features_in_)
        else:
            n_features = infer_feature_count(Path(model_path).parent)
        print(f"Inferred input features: {n_features}")
        try:
            initial_type = [('input', FloatTensorType([None, n_features]))]
            onx = convert_sklearn(model, initial_types=initial_type)
            onx.save(str(onnx_path))
            print(f"Saved ONNX to {onnx_path}")
        except Exception as ex:
            print("ONNX export failed:", ex)
            print("You may need to provide a sample numpy input or adjust initial_types.")
    except Exception as e:
        print("skl2onnx not available or export not implemented here.")
        print("Recommended command (once skl2onnx available):")
        print(f"python -c \"from skl2onnx import convert_sklearn; import joblib; m=joblib.load('{model_path}'); # build sample input X; onx=convert_sklearn(m, initial_types=[('input', FloatTensorType([None, N]))]); onx.save('{onnx_path}')\"")

    # Convert ONNX -> OpenVINO IR (if openvino-dev installed)
    try:
        import openvino
        print("OpenVINO available. Use Model Optimizer or ovc to convert ONNX to IR.")
        if onnx_path.exists():
            print(f"Converting {onnx_path} -> {xml_path}")
            run(f"python -m openvino.tools.mo --input_model {onnx_path} --output_dir {OUT_DIR}")
        else:
            print("ONNX not found; skip conversion until ONNX available")
    except Exception:
        print("OpenVINO not available. To install (recommended in Python 3.12 env):")
        print("pip install openvino openvino-dev")


def convert_pytorch(pt_path, name, sample_input_shape=(1,3,224,224)):
    onnx_path = OUT_DIR / f"{name}.onnx"
    xml_path = OUT_DIR / f"{name}.xml"
    try:
        import torch
        import onnx
        model = torch.load(pt_path, map_location='cpu')
        model.eval()
        import torch
        dummy = torch.randn(*sample_input_shape)
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=16)
        print(f"Exported ONNX to {onnx_path}")
    except Exception as e:
        print("PyTorch/ONNX export failed or not available:", e)
        print("Recommended: install torch and onnx, then export manually.")
    # Convert ONNX to OpenVINO IR
    try:
        import openvino
        if onnx_path.exists():
            run(f"python -m openvino.tools.mo --input_model {onnx_path} --output_dir {OUT_DIR}")
    except Exception:
        print("OpenVINO not installed. Install openvino-dev in a Python 3.12 env to enable conversion.")


def main():
    print("Starting conversion to OpenVINO IR — models in:", MODELS_DIR)
    for p in MODELS_DIR.iterdir():
        name = p.stem
        if p.suffix in ('.pkl', '.joblib'):
            convert_sklearn(p, name)
        elif p.suffix in ('.pt', '.pth'):
            convert_pytorch(p, name)
    print("Done. OpenVINO IR output directory:", OUT_DIR)

if __name__ == '__main__':
    main()
