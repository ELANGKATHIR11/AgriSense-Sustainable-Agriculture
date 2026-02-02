"""
Quantize the phi2-agriculture model to reduce size from ~5.3GB to ~2-3GB.
Uses 8-bit or 4-bit quantization with minimal quality loss.
"""

from pathlib import Path

# Paths
MODEL_DIR = Path(__file__).parent / "models" / "phi2-agriculture"
QUANTIZED_DIR = Path(__file__).parent / "models" / "phi2-agriculture-quantized"


def check_model_exists():
    """Check if the phi2 model exists"""
    if not MODEL_DIR.exists():
        print(f"❌ Model directory not found: {MODEL_DIR}")
        return False

    # Check for model files
    safetensors_files = list(MODEL_DIR.glob("*.safetensors"))
    bin_files = list(MODEL_DIR.glob("*.bin"))

    if not safetensors_files and not bin_files:
        print(f"❌ No model files found in {MODEL_DIR}")
        return False

    print("✓ Found model files:")
    for f in safetensors_files + bin_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.2f} MB")

    return True


def get_model_size(directory):
    """Calculate total size of model directory"""
    total_size = 0
    for file in Path(directory).rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size
    return total_size / (1024 * 1024)  # Convert to MB


"""
Quantization tools removed: this script previously performed bitsandbytes/
PyTorch/ONNX quantization which relies on GPU/NPU toolchains. To keep the
project CPU-only, quantization utilities have been removed.

If you need quantization in the future, reintroduce a vetted, optional tool
with clear install instructions and avoid automatic device_map usage.
"""

print("[INFO] Quantization tool removed (GPU/NPU features stripped).")

