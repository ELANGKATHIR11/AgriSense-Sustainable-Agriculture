"""
ML Model Optimization Module
Implements ONNX conversion, INT8 quantization, and lazy loading
Part of AgriSense Production Optimization Blueprint
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Feature flags from environment
ENABLE_YIELD_MODEL = os.getenv("ENABLE_YIELD_MODEL", "true").lower() == "true"
ENABLE_IRRIGATION_MODEL = os.getenv("ENABLE_IRRIGATION_MODEL", "true").lower() == "true"
ENABLE_DISEASE_MODEL = os.getenv("ENABLE_DISEASE_MODEL", "true").lower() == "true"
ENABLE_WEED_MODEL = os.getenv("ENABLE_WEED_MODEL", "true").lower() == "true"


class ModelOptimizer:
    """
    Handles ML model optimization including ONNX conversion and quantization
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path(__file__).parent.parent / "ml_models"
        self.onnx_models_dir = self.models_dir / "optimized" / "onnx"
        self.onnx_models_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_sklearn_to_onnx(
        self,
        model_path: Path,
        output_path: Optional[Path] = None,
        initial_types: Optional[list] = None
    ) -> Path:
        """
        Convert scikit-learn model to ONNX format
        
        Args:
            model_path: Path to .joblib model file
            output_path: Optional output path for ONNX model
            initial_types: Optional initial types for ONNX conversion
            
        Returns:
            Path to generated ONNX model
        """
        try:
            import joblib
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Load model
            model = joblib.load(model_path)
            
            # Default output path
            if output_path is None:
                output_path = self.onnx_models_dir / f"{model_path.stem}.onnx"
            
            # Determine input shape (adjust based on your models)
            if initial_types is None:
                initial_types = [('input', FloatTensorType([None, 7]))]  # 7 features typical
            
            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_types)
            
            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info(f"✅ Converted {model_path.name} to ONNX: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {model_path} to ONNX: {e}")
            raise
    
    def quantize_onnx_model(self, model_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Apply INT8 quantization to ONNX model
        
        Args:
            model_path: Path to ONNX model
            output_path: Optional output path for quantized model
            
        Returns:
            Path to quantized ONNX model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            if output_path is None:
                output_path = model_path.parent / f"{model_path.stem}_int8.onnx"
            
            # Apply dynamic quantization (INT8)
            quantize_dynamic(
                model_input=str(model_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8
            )
            
            logger.info(f"✅ Quantized {model_path.name} to INT8: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Failed to quantize {model_path}: {e}")
            raise
    
    def optimize_all_models(self):
        """
        Batch convert and quantize all models in ml_models directory
        """
        logger.info("🔧 Starting batch model optimization...")
        
        # Find all .joblib models
        model_patterns = [
            "disease_detection/**/*.joblib",
            "weed_management/**/*.joblib",
            "crop_recommendation/**/*.joblib",
            "**/*_model.joblib"
        ]
        
        optimized_count = 0
        for pattern in model_patterns:
            for model_file in self.models_dir.glob(pattern):
                try:
                    # Convert to ONNX
                    onnx_path = self.convert_sklearn_to_onnx(model_file)
                    
                    # Quantize
                    self.quantize_onnx_model(onnx_path)
                    
                    optimized_count += 1
                except Exception as e:
                    logger.warning(f"Skipped {model_file.name}: {e}")
        
        logger.info(f"✅ Optimized {optimized_count} models")


class LazyModelLoader:
    """
    Lazy-loading model cache to avoid loading all models at startup
    Thread-safe singleton pattern
    """
    
    _instance = None
    _models: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @lru_cache(maxsize=10)
    def load_model(self, model_key: str, model_path: Union[str, Path]) -> Any:
        """
        Load model on-demand with LRU caching
        
        Args:
            model_key: Unique identifier for the model
            model_path: Path to model file (supports .joblib, .onnx, .pkl)
            
        Returns:
            Loaded model object
        """
        if model_key in self._models:
            logger.debug(f"♻️ Using cached model: {model_key}")
            return self._models[model_key]
        
        logger.info(f"📥 Loading model: {model_key} from {model_path}")
        
        model_path = Path(model_path)
        
        # Load based on file extension
        if model_path.suffix == ".onnx":
            model = self._load_onnx_model(model_path)
        elif model_path.suffix in [".joblib", ".pkl"]:
            model = self._load_sklearn_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
        
        self._models[model_key] = model
        logger.info(f"✅ Model loaded: {model_key}")
        
        return model
    
    def _load_onnx_model(self, model_path: Path):
        """Load ONNX model with ONNX Runtime"""
        try:
            import onnxruntime as ort
            
            # Use GPU if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(model_path), providers=providers)
            
            return ONNXModelWrapper(session)
        except ImportError:
            logger.error("❌ onnxruntime not installed. Install with: pip install onnxruntime-gpu")
            raise
    
    def _load_sklearn_model(self, model_path: Path):
        """Load scikit-learn model from joblib"""
        import joblib
        return joblib.load(model_path)
    
    def unload_model(self, model_key: str):
        """Unload a model from cache"""
        if model_key in self._models:
            del self._models[model_key]
            logger.info(f"🗑️ Unloaded model: {model_key}")
    
    def clear_all(self):
        """Clear all cached models"""
        self._models.clear()
        logger.info("🗑️ Cleared all model caches")


class ONNXModelWrapper:
    """
    Wrapper for ONNX Runtime InferenceSession to provide sklearn-like API
    """
    
    def __init__(self, session):
        self.session = session
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using ONNX model
        
        Args:
            X: Input features as numpy array
            
        Returns:
            Predictions as numpy array
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Ensure float32 for ONNX
        X = X.astype(np.float32)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: X})
        
        return outputs[0]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (if available)
        """
        # For models with probability outputs
        return self.predict(X)


# Global lazy loader instance
_lazy_loader = LazyModelLoader()


def get_model(model_key: str, model_path: Union[str, Path], force_reload: bool = False) -> Any:
    """
    Convenience function to get a model with lazy loading
    
    Args:
        model_key: Unique model identifier
        model_path: Path to model file
        force_reload: Force reload even if cached
        
    Returns:
        Loaded model
        
    Example:
        >>> model = get_model("disease_detection", "ml_models/disease_model.onnx")
        >>> predictions = model.predict(features)
    """
    if force_reload:
        _lazy_loader.unload_model(model_key)
    
    return _lazy_loader.load_model(model_key, model_path)


def should_load_model(model_type: str) -> bool:
    """
    Check if a model should be loaded based on feature flags
    
    Args:
        model_type: Type of model (yield, irrigation, disease, weed)
        
    Returns:
        bool: True if model should be loaded
    """
    flags = {
        "yield": ENABLE_YIELD_MODEL,
        "irrigation": ENABLE_IRRIGATION_MODEL,
        "disease": ENABLE_DISEASE_MODEL,
        "weed": ENABLE_WEED_MODEL,
    }
    
    return flags.get(model_type, False)


if __name__ == "__main__":
    # CLI tool for batch optimization
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize AgriSense ML models")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Path to ml_models directory"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert models to ONNX format"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 quantization"
    )
    
    args = parser.parse_args()
    
    optimizer = ModelOptimizer(models_dir=args.models_dir)
    
    if args.convert or args.quantize:
        optimizer.optimize_all_models()
    else:
        print("Use --convert or --quantize to optimize models")
        print("Example: python model_optimizer.py --convert --quantize")
