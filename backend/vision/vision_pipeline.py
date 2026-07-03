import logging
import base64
from typing import Optional
from backend.vision.image_loader import load_image_from_base64, image_to_base64
from backend.vision.image_validator import (
    validate_image_integrity,
    check_image_resolution,
    estimate_blur,
)
from backend.vision.image_preprocessor import standardize_channels, resize_image
from backend.ollama_service import analyze_image_vlm

logger = logging.getLogger("AgriVisionPipeline")


async def process_and_analyze_image(
    image_base64: str,
    mode: str = "disease",
    apply_augmentation: bool = False,
    file_name: Optional[str] = None,
) -> dict:
    """
    Validates, standardizes, preprocesses, and executes SmolVLM analysis on an input image.
    """
    warnings = []

    # 1. Load image
    try:
        image = load_image_from_base64(image_base64)
    except Exception as e:
        logger.error(f"Failed to load base64 image: {e}")
        return {
            "success": False,
            "error": "Failed to decode base64 image input",
            "results": {},
            "warnings": ["Loading failed"],
        }

    # 2. Integrity and resolution checks
    is_valid, err_msg = validate_image_integrity(image)
    if not is_valid:
        return {
            "success": False,
            "error": err_msg,
            "results": {},
            "warnings": ["Integrity failed"],
        }

    if not check_image_resolution(image, 128, 128):
        warnings.append(
            "Resolution is lower than 128x128 threshold. Analysis precision may degrade."
        )

    # 3. Blur detection check
    is_blurry, blur_var = estimate_blur(image)
    if is_blurry:
        warnings.append(
            f"Image might be blurry (variance: {blur_var:.2f}). Consider recapturing in sharp lighting."
        )

    # 4. Standardize and resize
    image = standardize_channels(image)
    processed_image = resize_image(image, (448, 448))

    # 5. Convert back to base64 for Ollama
    final_b64 = image_to_base64(processed_image)
    image_bytes = base64.b64decode(final_b64)

    # 6. Execute model inference using SmolVLM
    try:
        inference_result = await analyze_image_vlm(
            image_bytes, mode=mode, file_name=file_name
        )

        # Structure the standard output contract
        return {
            "success": True,
            "analysis_type": mode,
            "confidence": float(inference_result.get("confidence", 90.0) / 100.0),
            "results": inference_result,
            "recommendations": inference_result.get("recommendations", []),
            "warnings": warnings,
        }
    except Exception as e:
        logger.error(f"SmolVLM analysis failed: {e}")
        return {
            "success": False,
            "error": f"Vision model inference failed: {str(e)}",
            "results": {},
            "warnings": warnings,
        }
