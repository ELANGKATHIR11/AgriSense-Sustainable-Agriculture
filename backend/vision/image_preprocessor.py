from PIL import Image
import numpy as np

def resize_image(image: Image.Image, target_size: tuple[int, int] = (448, 448)) -> Image.Image:
    """Resizes the PIL Image to target dimensions using Lanczos filter."""
    return image.resize(target_size, Image.Resampling.LANCZOS)

def normalize_pixel_intensities(image: Image.Image) -> np.ndarray:
    """Converts the PIL Image to a numpy array normalized between 0.0 and 1.0."""
    arr = np.array(image, dtype=np.float32)
    return arr / 255.0

def standardize_channels(image: Image.Image) -> Image.Image:
    """Converts images to RGB if they are RGBA, grayscale, or otherwise indexed."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image
