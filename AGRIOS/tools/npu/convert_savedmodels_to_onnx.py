"""
Convert all SavedModel dirs under `agrisense_app/backend/models` matching
`crop_recommendation_tf_npu*` to ONNX using `tf2onnx`.

Usage: .venv_py312_npu\Scripts\python.exe tools/npu/convert_savedmodels_to_onnx.py
"""
from pathlib import Path
import sys

ROOT = Path("agrisense_app/backend")
MODELS_DIR = ROOT / "models"

def main():
    try:
        import tf2onnx
    except Exception as e:
        print("tf2onnx not available:", e)
        sys.exit(1)

    candidates = list(MODELS_DIR.glob("crop_recommendation_tf_npu*"))
    if not candidates:
        print("No SavedModel candidates found in", MODELS_DIR)
        return

    for c in candidates:
        if (c / "saved_model.pb").exists():
            out_path = MODELS_DIR / "openvino_npu" / f"{c.name}.onnx"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                print(f"Converting {c} -> {out_path}")
                model_proto, external_tensor_storage = tf2onnx.convert.from_saved_model(str(c), opset=13)
                with open(out_path, "wb") as f:
                    f.write(model_proto.SerializeToString())
                print("Saved ONNX:", out_path)
            except Exception as e:
                print("Conversion failed for", c, "error:", e)
        else:
            print("Skipping non-SavedModel path", c)

if __name__ == "__main__":
    main()
