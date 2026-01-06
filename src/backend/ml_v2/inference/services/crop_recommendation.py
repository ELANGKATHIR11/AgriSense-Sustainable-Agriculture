"""
Crop Recommendation Service - Production inference wrapper.

Uses CatBoost or TF-DF models depending on deployment target.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CropPrediction:
    """Structured crop prediction result."""
    crop_name: str
    confidence: float
    top_alternatives: List[Tuple[str, float]]
    features_importance: Optional[Dict[str, float]] = None


class CropRecommendationService:
    """
    Production-ready crop recommendation service.
    
    Supports multiple backends:
        - CatBoost (server)
        - TFLite (edge)
        - ONNX (cross-platform)
    
    Usage:
        service = CropRecommendationService()
        result = service.predict({
            "nitrogen": 90, "phosphorus": 42, "potassium": 43,
            "temperature": 20.87, "humidity": 82.0, "ph": 6.5,
            "rainfall": 202.9, "soil_type": "Loamy"
        })
    """
    
    def __init__(
        self, 
        model_path: Optional[Path] = None,
        backend: str = "auto"  # auto, catboost, onnx, tflite
    ):
        """Initialize service with model."""
        self.backend = backend
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        
        if model_path:
            self._load_model(model_path)
        else:
            self._load_default_model()
    
    def _load_default_model(self):
        """Load the default production model."""
        models_dir = Path(__file__).parent.parent.parent / "models" / "group_a_tabular"
        
        # Try CatBoost first, then ONNX, then TFLite
        catboost_path = models_dir / "catboost" / "crop_recommendation_v2.cbm"
        onnx_path = models_dir / "catboost" / "crop_recommendation_v2.onnx"
        tflite_path = models_dir / "tfdf" / "crop_recommendation_edge.tflite"
        
        if catboost_path.exists() and self.backend in ["auto", "catboost"]:
            self._load_catboost(catboost_path)
        elif onnx_path.exists() and self.backend in ["auto", "onnx"]:
            self._load_onnx(onnx_path)
        elif tflite_path.exists() and self.backend in ["auto", "tflite"]:
            self._load_tflite(tflite_path)
        else:
            logger.warning("No production model found. Service will return mock predictions.")
            self.model = None
    
    def _load_catboost(self, path: Path):
        """Load CatBoost model."""
        try:
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier()
            self.model.load_model(str(path))
            self.backend = "catboost"
            logger.info(f"Loaded CatBoost model from {path}")
        except ImportError:
            logger.error("CatBoost not installed. Install with: pip install catboost")
    
    def _load_onnx(self, path: Path):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(str(path))
            self.backend = "onnx"
            logger.info(f"Loaded ONNX model from {path}")
        except ImportError:
            logger.error("ONNX Runtime not installed. Install with: pip install onnxruntime")
    
    def _load_tflite(self, path: Path):
        """Load TFLite model."""
        try:
            import tflite_runtime.interpreter as tflite
            self.model = tflite.Interpreter(model_path=str(path))
            self.model.allocate_tensors()
            self.backend = "tflite"
            logger.info(f"Loaded TFLite model from {path}")
        except ImportError:
            try:
                import tensorflow as tf
                self.model = tf.lite.Interpreter(model_path=str(path))
                self.model.allocate_tensors()
                self.backend = "tflite"
            except ImportError:
                logger.error("TFLite not available. Install tensorflow or tflite-runtime")
    
    def _load_model(self, path: Path):
        """Load model from explicit path."""
        suffix = path.suffix.lower()
        if suffix == ".cbm":
            self._load_catboost(path)
        elif suffix == ".onnx":
            self._load_onnx(path)
        elif suffix == ".tflite":
            self._load_tflite(path)
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
    
    def predict(
        self, 
        features: Dict[str, Any],
        return_top_k: int = 3,
        return_importance: bool = False
    ) -> CropPrediction:
        """
        Predict recommended crop from input features.
        
        Args:
            features: Dictionary of input features
            return_top_k: Number of alternative crops to return
            return_importance: Whether to include feature importance
            
        Returns:
            CropPrediction with crop name, confidence, and alternatives
        """
        if self.model is None:
            # Return mock prediction for testing
            return CropPrediction(
                crop_name="Rice",
                confidence=0.85,
                top_alternatives=[("Wheat", 0.10), ("Maize", 0.05)]
            )
        
        # Prepare features
        feature_array = self._prepare_features(features)
        
        # Run inference based on backend
        if self.backend == "catboost":
            return self._predict_catboost(feature_array, return_top_k, return_importance)
        elif self.backend == "onnx":
            return self._predict_onnx(feature_array, return_top_k)
        elif self.backend == "tflite":
            return self._predict_tflite(feature_array, return_top_k)
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dict to array."""
        # Define expected feature order
        expected_features = [
            "nitrogen", "phosphorus", "potassium", "temperature",
            "humidity", "ph", "rainfall", "soil_type", "season"
        ]
        
        values = []
        for feat in expected_features:
            if feat in features:
                values.append(features[feat])
            else:
                values.append(0)  # Default value
        
        return np.array([values])
    
    def _predict_catboost(
        self, 
        features: np.ndarray, 
        top_k: int,
        return_importance: bool
    ) -> CropPrediction:
        """Run CatBoost prediction."""
        proba = self.model.predict_proba(features)[0]
        top_indices = np.argsort(proba)[::-1][:top_k]
        
        class_names = self.model.classes_
        
        prediction = CropPrediction(
            crop_name=class_names[top_indices[0]],
            confidence=float(proba[top_indices[0]]),
            top_alternatives=[
                (class_names[idx], float(proba[idx]))
                for idx in top_indices[1:]
            ]
        )
        
        if return_importance:
            importance = self.model.get_feature_importance()
            feature_names = self.model.feature_names_
            prediction.features_importance = dict(zip(feature_names, importance))
        
        return prediction
    
    def _predict_onnx(self, features: np.ndarray, top_k: int) -> CropPrediction:
        """Run ONNX prediction."""
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: features.astype(np.float32)})
        
        proba = outputs[1][0]  # Assuming second output is probabilities
        label = outputs[0][0]  # First output is label
        
        top_indices = np.argsort(proba)[::-1][:top_k]
        
        return CropPrediction(
            crop_name=str(label),
            confidence=float(max(proba)),
            top_alternatives=[
                (str(idx), float(proba[idx]))
                for idx in top_indices[1:]
            ]
        )
    
    def _predict_tflite(self, features: np.ndarray, top_k: int) -> CropPrediction:
        """Run TFLite prediction."""
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        
        self.model.set_tensor(
            input_details[0]['index'], 
            features.astype(np.float32)
        )
        self.model.invoke()
        
        output = self.model.get_tensor(output_details[0]['index'])[0]
        top_indices = np.argsort(output)[::-1][:top_k]
        
        return CropPrediction(
            crop_name=str(top_indices[0]),
            confidence=float(output[top_indices[0]]),
            top_alternatives=[
                (str(idx), float(output[idx]))
                for idx in top_indices[1:]
            ]
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Return service health status."""
        return {
            "service": "crop_recommendation",
            "status": "healthy" if self.model else "degraded",
            "backend": self.backend,
            "model_loaded": self.model is not None
        }
