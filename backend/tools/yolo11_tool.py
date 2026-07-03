# backend/tools/yolo11_tool.py
"""Tool for YOLO11 VLM image analysis via Ollama.

Expected payload:
{
  "imageBase64": "...",
  "mode": "disease" | "weed" (optional, defaults to "disease")
}
"""
import base64
from backend.ollama_service import analyze_image_vlm

async def detect_image(payload: dict) -> dict:
    """Detect objects/diseases in an image using the SmolVLM model.
    Returns the JSON result from the VLM.
    """
    image_b64 = payload.get("imageBase64")
    if not image_b64:
        raise ValueError("imageBase64 is required")
    mode = payload.get("mode", "disease")
    image_bytes = base64.b64decode(image_b64)
    # Use the default VLM model (riven/smolvlm:latest)
    result = await analyze_image_vlm(image_bytes, mode=mode, vlm_model="riven/smolvlm:latest")
    return result
