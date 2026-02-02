import logging
import os

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logger = logging.getLogger("AgriSense-AI")

# Model configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# The filename must match what is downloaded in download_model.py
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# MODEL_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf" # Uncomment if using Phi-3


class LLMEngine:
    def __init__(self, model_name: str = None):
        self.model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        self.llm = None
        self._load_model()

    def _load_model(self):
        if not Llama:
            logger.error("llama-cpp-python is not installed. Please install it.")
            return

        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at {self.model_path}")
            logger.warning("Please run 'python download_model.py' to fetch the model.")
            return

        try:
            logger.info(f"Loading local GGUF model from {self.model_path}...")
            # n_ctx=2048 for context window, n_gpu_layers=0 for CPU only (increase if you have GPU)
            # verbose=False to reduce spam
            self.llm = Llama(
                model_path=self.model_path, n_ctx=2048, n_gpu_layers=0, verbose=False
            )
            logger.info("✅ Native GGUF Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generates a direct response from the local GGUF model.
        """
        if not self.llm:
            return "Based on standard agricultural guidelines, please monitor soil moisture. (Model offline)"

        # Simple prompt templating for Chat models (TinyLlama/Phi-3 style)
        # Adjust template based on exact model used. This is a generic chat template.
        full_prompt = ""
        if system_prompt:
            full_prompt += f"<|system|>\n{system_prompt}</s>\n"
        full_prompt += f"<|user|>\n{prompt}</s>\n<|assistant|>\n"

        try:
            output = self.llm(
                full_prompt, max_tokens=512, stop=["</s>"], echo=False, temperature=0.7
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble reasoning right now."

    def stream_response(self, prompt: str, system_prompt: str = None):
        """
        Streams response from LLM (for "Thinking..." effect).
        """
        if not self.llm:
            yield "Model offline."
            return

        full_prompt = ""
        if system_prompt:
            full_prompt += f"<|system|>\n{system_prompt}</s>\n"
        full_prompt += f"<|user|>\n{prompt}</s>\n<|assistant|>\n"

        try:
            stream = self.llm(
                full_prompt,
                max_tokens=512,
                stop=["</s>"],
                stream=True,
                echo=False,
                temperature=0.7,
            )
            for chunk in stream:
                yield chunk["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield " Error."
