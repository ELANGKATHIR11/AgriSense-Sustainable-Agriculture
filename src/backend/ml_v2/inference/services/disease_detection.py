"""
Disease Detection Service - ConvNeXt V2 inference wrapper.

Provides plant disease detection from leaf images.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DiseasePrediction:
    """Structured disease prediction result."""
    disease_name: str
    confidence: float
    is_healthy: bool
    severity: Optional[str] = None  # mild, moderate, severe
    treatment_recommendations: Optional[List[str]] = None
    top_alternatives: Optional[List[Tuple[str, float]]] = None


class DiseaseDetectionService:
    """
    Production disease detection service using ConvNeXt V2 Nano.
    
    Usage:
        service = DiseaseDetectionService()
        result = service.predict(image_array)
    """
    
    DISEASE_CLASSES = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
        "Apple___healthy", "Corn___Cercospora_leaf_spot",
        # ... 38 PlantVillage classes
    ]
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize with ONNX model."""
        self.model = None
        self.input_size = 384
        
        if model_path:
            self._load_model(model_path)
        else:
            self._load_default_model()
    
    def _load_default_model(self):
        """Load default production model."""
        models_dir = Path(__file__).parent.parent.parent / "models" / "group_b_vision"
        onnx_path = models_dir / "convnext" / "disease_detector_v2.onnx"
        
        if onnx_path.exists():
            self._load_model(onnx_path)
        else:
            logger.warning("Disease detection model not found.")
    
    def _load_model(self, path: Path):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(str(path))
            logger.info(f"Loaded disease model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ConvNeXt V2."""
        import cv2
        
        # Resize to input size
        if image.shape[:2] != (self.input_size, self.input_size):
            image = cv2.resize(image, (self.input_size, self.input_size))
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image / 255.0 - mean) / std
        
        # HWC -> CHW, add batch dim
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0).astype(np.float32)
        
        return image
    
    def predict(
        self, 
        image: np.ndarray,
        return_top_k: int = 3
    ) -> DiseasePrediction:
        """
        Predict disease from leaf image.
        
        Args:
            image: RGB image array (H, W, 3)
            return_top_k: Number of alternative predictions
            
        Returns:
            DiseasePrediction with disease info
        """
        if self.model is None:
            return DiseasePrediction(
                disease_name="Unknown",
                confidence=0.0,
                is_healthy=True,
                severity=None
            )
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_tensor})
        
        proba = outputs[0][0]
        top_indices = np.argsort(proba)[::-1][:return_top_k]
        
        predicted_class = self.DISEASE_CLASSES[top_indices[0]]
        confidence = float(proba[top_indices[0]])
        
        return DiseasePrediction(
            disease_name=predicted_class,
            confidence=confidence,
            is_healthy="healthy" in predicted_class.lower(),
            severity=self._estimate_severity(confidence),
            top_alternatives=[
                (self.DISEASE_CLASSES[idx], float(proba[idx]))
                for idx in top_indices[1:]
            ]
        )
    
    def _estimate_severity(self, confidence: float) -> str:
        """Estimate disease severity from confidence."""
        if confidence > 0.9:
            return "severe"
        elif confidence > 0.7:
            return "moderate"
        else:
            return "mild"
