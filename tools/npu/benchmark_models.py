"""
Benchmark trained models (OpenVINO NPU and sklearn joblib fallbacks).

Usage: python tools/npu/benchmark_models.py

This script will:
- Load OpenVINO IR/ONNX models from `agrisense_app/backend/models/openvino_npu` and run inference on `NPU`.
- Load sklearn joblib models in `agrisense_app/backend/models/` and run numpy-based inference to compare.
"""
import time
from pathlib import Path
import numpy as np
import joblib

ROOT = Path("agrisense_app/backend/models")
OPENVINO_DIR = ROOT / "openvino_npu"

def benchmark_sklearn(model_path, scaler_path=None, n_runs=2000, batch_size=32):
    print(f"\n-- Benchmark sklearn model: {model_path.name}")
    model = joblib.load(model_path)
    scaler = None
    if scaler_path and scaler_path.exists():
        scaler = joblib.load(scaler_path)

    n_features = getattr(scaler, "n_features_in_", 7) if scaler is not None else 7
    # synthetic batch
    X = np.random.rand(batch_size, n_features).astype(np.float32)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    # warmup
    for _ in range(5):
        _ = model.predict(X)

    t0 = time.time()
    iters = max(1, n_runs // batch_size)
    for _ in range(iters):
        _ = model.predict(X)
    elapsed = time.time() - t0
    total_samples = iters * batch_size
    print(f"   Samples: {total_samples}, Time: {elapsed:.4f}s, Throughput: {total_samples/elapsed:.1f} samples/s")


def benchmark_openvino(model_folder: Path, device: str = "NPU", n_runs: int = 2000, batch_size: int = 32):
    print(f"\n-- Benchmark OpenVINO model: {model_folder.name} on {device}")
    try:
        from openvino.runtime import Core
        core = Core()
    except Exception as e:
        print("OpenVINO not available:", e)
        return

    # handle if caller passed a file path directly
    if model_folder.is_file() and model_folder.suffix == ".onnx":
        model_path = model_folder
    else:
        # try to find XML or ONNX inside the folder
        xml = None
        onnx = None
        try:
            for f in model_folder.iterdir():
                if f.suffix == ".xml":
                    xml = f
                if f.suffix == ".onnx":
                    onnx = f
        except Exception:
            print("No XML/ONNX found in", model_folder)
            return

        model_path = xml or onnx
        if model_path is None:
            print("No XML/ONNX found in", model_folder)
            return

    try:
        compiled = core.compile_model(str(model_path), device)
    except Exception as e:
        print("Failed to compile model on device:", e)
        return

    input_ports = compiled.inputs
    # build sample input based on shapes
    shape = list(input_ports[0].shape)
    # replace dynamic dims with batch_size
    for i, s in enumerate(shape):
        if s is None or s == 0:
            shape[i] = batch_size
    X = np.random.rand(*shape).astype(np.float32)

    # warmup
    for _ in range(5):
        compiled([X])

    t0 = time.time()
    iters = max(1, n_runs // batch_size)
    for _ in range(iters):
        compiled([X])
    elapsed = time.time() - t0
    total_samples = iters * batch_size
    print(f"   Samples: {total_samples}, Time: {elapsed:.4f}s, Throughput: {total_samples/elapsed:.1f} samples/s")


def main():
    print("Starting model benchmarks...")

    # Benchmark sklearn models
    scaler_path = ROOT / "crop_scaler.joblib"
    for m in ROOT.glob("*_rf_npu.joblib"):
        benchmark_sklearn(m, scaler_path)
    for m in ROOT.glob("*_gb_npu.joblib"):
        benchmark_sklearn(m, scaler_path)

    # Benchmark OpenVINO models found in openvino_npu
    if OPENVINO_DIR.exists():
        # handle both directories (XML+bin) and standalone ONNX files
        for item in OPENVINO_DIR.iterdir():
            if item.is_dir():
                benchmark_openvino(item, device="NPU")
            elif item.suffix == ".onnx":
                # create a temporary folder-like wrapper
                benchmark_openvino(OPENVINO_DIR / item.name, device="NPU")
    else:
        print("No OpenVINO models directory found:", OPENVINO_DIR)


if __name__ == "__main__":
    main()
