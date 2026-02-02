from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models/phi2-agriculture"

print(f"Testing model load from: {MODEL_PATH}")

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    print("✅ Tokenizer loaded")

    print("Loading model (CPU)...")
    # minimal load
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print(f"✅ Model loaded. Params: {model.num_parameters()}")

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
