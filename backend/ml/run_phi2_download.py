from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI2_DIR = Path(r"f:\AGRISENSEFULL-STACK\backend\ml\models\phi2-agriculture")
PHI2_DIR.mkdir(exist_ok=True, parents=True)

print("🤖 Downloading Phi-2 from HuggingFace...")
print("=" * 70)
print("This will download ~2.5 GB - please be patient (5-10 minutes)")
print("=" * 70 + "\n")

try:
    print("Step 1/2: Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2", trust_remote_code=True
    )
    tokenizer.save_pretrained(PHI2_DIR)
    print("✅ Tokenizer downloaded and saved\n")

    print("Step 2/2: Downloading model (this is the large file ~2.5 GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", trust_remote_code=True, torch_dtype=torch.float16
    )

    print("💾 Saving model to disk...")
    model.save_pretrained(PHI2_DIR)

    print("\n" + "=" * 70)
    print("✅ Phi-2 Successfully Downloaded!")
    print("=" * 70)
    print(f"\nSaved to: {PHI2_DIR}")
    print("\nModel is ready to use!")
    print("\nTest it now:")
    print("   python native_agricultural_advisor.py")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nIf download failed, check internet connection and try again")
