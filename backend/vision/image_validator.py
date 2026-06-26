import numpy as np
from PIL import Image

def validate_image_integrity(image: Image.Image) -> tuple[bool, str]:
    """
    Checks if an image is corrupted or empty.
    Returns (is_valid, error_message).
    """
    try:
        image.verify()
        return True, ""
    except Exception as e:
        return False, f"Image file corrupted or invalid: {str(e)}"

def estimate_blur(image: Image.Image, threshold: float = 10.0) -> tuple[bool, float]:
    """
    Estimates blur using the variance of Laplacian of the grayscale image.
    If OpenCV is not available, we use a numpy-based fallback.
    Returns (is_blurry, variance).
    """
    try:
        # Fallback using numpy gradients to avoid strict opencv-python dependency
        gray = image.convert("L")
        img_arr = np.array(gray, dtype=np.float64)
        
        # Calculate Laplacian approximation using simple finite difference kernels
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        # Simple convolution
        from scipy.signal import convolve2d
        laplacian = convolve2d(img_arr, laplacian_kernel, mode='same')
        
        variance = laplacian.var()
        is_blurry = variance < threshold
        return is_blurry, float(variance)
    except Exception:
        # If scipy is missing, calculate variance of simple differences
        img_arr = np.array(image.convert("L"), dtype=np.float64)
        dx = np.diff(img_arr, axis=1)
        dy = np.diff(img_arr, axis=0)
        variance = float(np.var(dx) + np.var(dy))
        # Scaled threshold
        return variance < 8.0, variance

def check_image_resolution(image: Image.Image, min_width: int = 128, min_height: int = 128) -> bool:
    """Verifies that the image resolution exceeds the minimum dimension thresholds."""
    width, height = image.size
    return width >= min_width and height >= min_height
