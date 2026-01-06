"""
ML Model Security & Validation
Ensures safe inference with validation of inputs and outputs
"""
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class ModelSecurityValidator:
    """
    Validates ML model inputs and outputs for security and correctness.
    Prevents adversarial attacks and validates model behavior.
    """

    def __init__(self, model_name: str, expected_signature: Optional[str] = None):
        self.model_name = model_name
        self.expected_signature = expected_signature
        self.max_input_size = 10000  # Maximum input size in KB
        self.confidence_threshold = 0.5
        self.suspicious_outputs = []

    def verify_model_integrity(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """
        Verify model integrity using signature verification.
        Prevents model tampering.
        """
        if self.expected_signature is None:
            logger.warning(f"No signature available for {self.model_name}")
            return True, None

        try:
            # Calculate SHA256 hash of model file
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            calculated_signature = sha256_hash.hexdigest()

            if calculated_signature != self.expected_signature:
                logger.error(
                    f"Model signature mismatch for {self.model_name}. "
                    f"Expected: {self.expected_signature}, Got: {calculated_signature}"
                )
                return False, "Model signature verification failed"

            logger.info(f"Model {self.model_name} signature verified")
            return True, None

        except Exception as e:
            logger.error(f"Error verifying model signature: {e}")
            return False, str(e)

    def validate_input(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate model input for safety and correctness.
        Returns: (is_valid, error_message)
        """
        # Check input type
        if not isinstance(data, (dict, list, tuple)):
            return False, "Input must be dict, list, or tuple"

        # Check input size
        try:
            input_str = json.dumps(data)
            input_size_kb = len(input_str.encode()) / 1024
            if input_size_kb > self.max_input_size:
                return (
                    False,
                    f"Input size ({input_size_kb:.1f} KB) exceeds maximum ({self.max_input_size} KB)",
                )
        except Exception as e:
            return False, f"Failed to serialize input: {e}"

        # Validate numeric ranges if dict
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    if not (-1e10 <= value <= 1e10):
                        return False, f"Value for {key} out of reasonable range: {value}"

                    # Check for NaN and Inf
                    if isinstance(value, float):
                        if not (-float("inf") < value < float("inf")):
                            return False, f"Invalid float value for {key}: {value}"

        return True, None

    def validate_output(self, output: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate model output for sanity.
        Returns: (is_valid, error_message)
        """
        # Check output type
        if not isinstance(output, (dict, list, int, float, str)):
            return False, f"Unexpected output type: {type(output)}"

        # Validate numeric outputs
        if isinstance(output, (int, float)):
            # Check for NaN and Inf
            if isinstance(output, float):
                if not (-float("inf") < output < float("inf")):
                    return False, f"Invalid output value (NaN or Inf): {output}"

            # Check for extreme values
            if not (-1e10 <= output <= 1e10):
                logger.warning(f"Output value outside typical range: {output}")

        # Validate dict outputs
        elif isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, float):
                    if not (-float("inf") < value < float("inf")):
                        return False, f"Invalid value in output[{key}]: {value}"

            # Check for confidence/probability scores
            if "confidence" in output or "probability" in output:
                score_key = "confidence" if "confidence" in output else "probability"
                score = output[score_key]
                if not (0.0 <= score <= 1.0):
                    return (
                        False,
                        f"Invalid {score_key} score (should be 0-1): {score}",
                    )

        return True, None

    def is_suspicious_output(self, output: Any, threshold: float = 0.95) -> bool:
        """
        Detect suspicious outputs that might indicate adversarial attack.
        E.g., predictions with extremely high confidence (>0.99)
        """
        if isinstance(output, dict):
            if "confidence" in output and output["confidence"] > threshold:
                logger.warning(f"Suspicious high confidence prediction: {output}")
                return True

            if "probability" in output and output["probability"] > threshold:
                logger.warning(f"Suspicious high probability prediction: {output}")
                return True

        return False


class AdversarialAttackDetection:
    """
    Detect and prevent adversarial attacks on ML models.
    """

    @staticmethod
    def detect_input_perturbation(
        original: List[float], perturbed: List[float], threshold: float = 0.01
    ) -> bool:
        """
        Detect if input has been adversarially perturbed.
        Compares original and perturbed inputs.
        """
        if len(original) != len(perturbed):
            return False

        # Calculate L2 distance
        distance = sum((a - b) ** 2 for a, b in zip(original, perturbed)) ** 0.5

        if distance > threshold:
            logger.warning(f"Possible adversarial perturbation detected: {distance}")
            return True

        return False

    @staticmethod
    def detect_gradient_attack(gradients: List[float], threshold: float = 1.0) -> bool:
        """
        Detect potential gradient-based attack.
        Extremely large gradients might indicate attack.
        """
        max_gradient = max(abs(g) for g in gradients) if gradients else 0

        if max_gradient > threshold:
            logger.warning(f"Large gradients detected (possible attack): {max_gradient}")
            return True

        return False


class ModelInputSanitizer:
    """
    Sanitize inputs before ML model inference.
    """

    @staticmethod
    def sanitize_sensor_data(
        data: Dict[str, float],
        valid_ranges: Dict[str, Tuple[float, float]],
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, float]]]:
        """
        Sanitize sensor input with validation against expected ranges.
        Returns: (is_valid, error_message, sanitized_data)
        """
        sanitized = {}

        for key, value in data.items():
            if key not in valid_ranges:
                return False, f"Unknown sensor field: {key}", None

            min_val, max_val = valid_ranges[key]

            # Check range
            if not (min_val <= value <= max_val):
                return (
                    False,
                    f"Value {key}={value} outside range [{min_val}, {max_val}]",
                    None,
                )

            # Check for NaN/Inf
            if isinstance(value, float):
                if not (-float("inf") < value < float("inf")):
                    return False, f"Invalid value for {key}: {value}", None

            sanitized[key] = value

        return True, None, sanitized

    @staticmethod
    def clip_to_range(
        value: float, min_val: float, max_val: float, clip: bool = True
    ) -> float:
        """
        Clip value to valid range (if clip=True) or return error.
        """
        if not clip:
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Value {value} outside range [{min_val}, {max_val}]"
                )
            return value

        return max(min_val, min(value, max_val))


class ModelOutputSanitizer:
    """
    Sanitize outputs before returning to users.
    """

    @staticmethod
    def format_prediction(
        prediction: Any, output_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Format model output safely.
        """
        if output_type == "classification":
            return {
                "class": prediction.get("class"),
                "confidence": float(prediction.get("confidence", 0.0)),
                "alternatives": prediction.get("alternatives", []),
                "timestamp": prediction.get("timestamp"),
            }

        elif output_type == "regression":
            return {
                "value": float(prediction.get("value", 0.0)),
                "uncertainty": float(prediction.get("uncertainty", 0.0)),
                "timestamp": prediction.get("timestamp"),
            }

        elif output_type == "segmentation":
            return {
                "mask_url": prediction.get("mask_url"),
                "classes": prediction.get("classes", []),
                "confidence": float(prediction.get("confidence", 0.0)),
            }

        else:
            return prediction

    @staticmethod
    def validate_predictions(predictions: List[Dict], min_count: int = 1) -> bool:
        """
        Validate batch predictions.
        """
        if len(predictions) < min_count:
            logger.warning(f"Insufficient predictions: {len(predictions)}")
            return False

        for i, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                logger.warning(f"Invalid prediction format at index {i}")
                return False

        return True
