from huggingface_hub import hf_hub_download
import os

# Configuration
MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# Alternate (Better Quality, ~2.4GB):
# MODEL_REPO = "imartinez/Phi-3-mini-4k-instruct-GGUF"
# MODEL_FILE = "Phi-3-mini-4k-instruct-q4.gguf"

DEST_DIR = os.path.join(os.path.dirname(__file__), "models")


def download_model():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    print(f"Downloading {MODEL_FILE} from {MODEL_REPO}...")
    print(f"This is a one-time download (~640 MB). Please wait.")

    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=DEST_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"\n✅ Model downloaded successfully to: {model_path}")
        print("You can now restart the AI service.")
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")


if __name__ == "__main__":
    download_model()
