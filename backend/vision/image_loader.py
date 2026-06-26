import base64
import io
from PIL import Image

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Loads a PIL Image from raw bytes."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def load_image_from_base64(base64_str: str) -> Image.Image:
    """Loads a PIL Image from a base64 encoded string."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_bytes = base64.b64decode(base64_str)
    return load_image_from_bytes(image_bytes)

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Converts a PIL Image back to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
