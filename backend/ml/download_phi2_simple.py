"""
Direct Phi-2 Download Script (Simplified)
Downloads microsoft/phi-2 without complex quantization
"""

from pathlib import Path

print("🤖 Downloading Phi-2 Model")
print("=" * 70)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
PHI2_DIR = MODELS_DIR / "phi2-agriculture"
PHI2_DIR.mkdir(exist_ok=True, parents=True)

print(f"Target directory: {PHI2_DIR}\n")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("✅ Dependencies available\n")

    # Check if already downloaded
    if (PHI2_DIR / "config.json").exists():
        print("✅ Phi-2 already exists!")
        print(f"Location: {PHI2_DIR}")
        print("\nModel is ready to use!")
        exit(0)

    print("📥 Downloading microsoft/phi-2...")
    print("   This will take 5-10 minutes (~2.5-5 GB)")
    print("   Please be patient...\n")

    # Download tokenizer
    print("Step 1/2: Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2", trust_remote_code=True
    )
    tokenizer.save_pretrained(PHI2_DIR)
    print("✅ Tokenizer saved\n")

    # Download model
    print("Step 2/2: Downloading model (this is the large file)...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Half precision to save space
        low_cpu_mem_usage=True,
    )

    print("💾 Saving model to disk...")
    model.save_pretrained(PHI2_DIR)

    print("\n" + "=" * 70)
    print("✅ Phi-2 Download Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {PHI2_DIR}")
    print("\nNext: Test the chatbot")
    print("   python native_agricultural_advisor.py")

except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("\nPlease install: pip install transformers torch")
    exit(1)
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("\nTry running again or check internet connection")
    exit(1)
