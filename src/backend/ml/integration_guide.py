"""
ML Model Security Integration Guide

This module integrates ML model security validation into inference pipelines.
Prevents adversarial attacks, validates inputs/outputs, ensures model integrity.
"""

from typing import Any, Dict, Optional, Tuple
from backend.ml.security import ModelSecurityValidator, ModelInputSanitizer, ModelOutputSanitizer
import time
import logging

logger = logging.getLogger(__name__)


class InferenceSecurityWrapper:
    """
    Wraps ML model inference with security validation.
    Usage:
        wrapper = InferenceSecurityWrapper("crop_recommendation_model")
        result = wrapper.predict(features, model_func=model.predict)
    """

    def __init__(
        self,
        model_name: str,
        expected_signature: Optional[str] = None,
        timeout_seconds: int = 30,
    ):
        self.model_name = model_name
        self.validator = ModelSecurityValidator(model_name, expected_signature)
        self.timeout_seconds = timeout_seconds
        self.inference_count = 0
        self.blocked_count = 0

    def verify_model(self, model_path: str) -> bool:
        """Verify model integrity before inference."""
        is_valid, error = self.validator.verify_model_integrity(model_path)
        if not is_valid:
            logger.error(f"Model verification failed: {error}")
            return False
        return True

    def predict(
        self,
        features: Dict[str, Any],
        model_func,
        input_validation_rules: Optional[Dict] = None,
        output_type: str = "classification",
    ) -> Optional[Dict]:
        """
        Secure inference wrapper with validation.
        
        Args:
            features: Input features for model
            model_func: Callable that performs inference
            input_validation_rules: Dict of {field: (min, max)} for range checking
            output_type: Type of output ('classification', 'regression', 'segmentation')
            
        Returns:
            Validated model output or None if validation fails
        """
        self.inference_count += 1

        # Step 1: Validate input
        is_valid, error = self.validator.validate_input(features)
        if not is_valid:
            logger.warning(f"Input validation failed: {error}")
            self.blocked_count += 1
            return None

        # Step 2: Sanitize input
        if input_validation_rules:
            is_valid, error, sanitized = ModelInputSanitizer.sanitize_sensor_data(
                features, input_validation_rules
            )
            if not is_valid:
                logger.warning(f"Input sanitization failed: {error}")
                self.blocked_count += 1
                return None
            features = sanitized

        # Step 3: Run inference with timeout
        try:
            start_time = time.time()
            output = model_func(features)
            elapsed = time.time() - start_time

            if elapsed > self.timeout_seconds:
                logger.error(f"Inference timeout (>{self.timeout_seconds}s)")
                self.blocked_count += 1
                return None

            logger.info(f"Inference completed in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            self.blocked_count += 1
            return None

        # Step 4: Validate output
        is_valid, error = self.validator.validate_output(output)
        if not is_valid:
            logger.error(f"Output validation failed: {error}")
            self.blocked_count += 1
            return None

        # Step 5: Detect suspicious outputs
        if self.validator.is_suspicious_output(output):
            logger.warning(f"Suspicious output detected for {self.model_name}")

        # Step 6: Format and return sanitized output
        formatted_output = ModelOutputSanitizer.format_prediction(output, output_type)

        return formatted_output

    def get_stats(self) -> Dict:
        """Get inference statistics."""
        block_rate = (self.blocked_count / self.inference_count * 100) if self.inference_count > 0 else 0
        return {
            "model": self.model_name,
            "total_inferences": self.inference_count,
            "blocked_requests": self.blocked_count,
            "block_rate_percent": block_rate,
            "acceptance_rate_percent": 100 - block_rate,
        }


# Example integration with FastAPI endpoint
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])

# Initialize wrappers for each model
crop_wrapper = InferenceSecurityWrapper("crop_recommendation_model")
disease_wrapper = InferenceSecurityWrapper("disease_detection_model")

class SensorReadingRequest(BaseModel):
    device_id: str
    temperature: float
    humidity: float
    soil_moisture: float
    soil_ph: float
    nitrogen: float
    phosphorus: float
    potassium: float

@router.post("/crop-recommendation")
async def get_crop_recommendation(request: SensorReadingRequest):
    # Extract features
    features = {
        "temperature": request.temperature,
        "humidity": request.humidity,
        "soil_moisture": request.soil_moisture,
        "soil_ph": request.soil_ph,
        "nitrogen": request.nitrogen,
        "phosphorus": request.phosphorus,
        "potassium": request.potassium,
    }
    
    # Define valid ranges
    valid_ranges = {
        "temperature": (-20, 60),
        "humidity": (0, 100),
        "soil_moisture": (0, 100),
        "soil_ph": (0, 14),
        "nitrogen": (0, 1000),
        "phosphorus": (0, 1000),
        "potassium": (0, 1000),
    }
    
    # Run secure inference
    result = crop_wrapper.predict(
        features,
        model_func=crop_model.predict,
        input_validation_rules=valid_ranges,
        output_type="classification"
    )
    
    if result is None:
        raise HTTPException(status_code=400, detail="Invalid input or inference failed")
    
    return result

@router.get("/ml-stats")
async def get_ml_stats():
    return {
        "crop_recommendation": crop_wrapper.get_stats(),
        "disease_detection": disease_wrapper.get_stats(),
    }
"""
