"""
Inference Utilities for AgriSense ML Models
Load models, make predictions, handle feature scaling
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set up paths
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
ENCODERS_DIR = DATA_DIR / "encoders"


class ModelInference:
    """Handle ML model inference and predictions"""
    
    def __init__(self):
        """Load all models and encoders"""
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_metrics = {}
        
        self._load_models()
        self._load_encoders()
        self._load_metrics()
    
    def _load_models(self):
        """Load trained ML models"""
        model_files = {
            'crop_recommendation': 'crop_recommendation_model.pkl',
            'crop_type_classification': 'crop_type_classification_model.pkl',
            'growth_duration': 'growth_duration_model.pkl',
            'water_requirement': 'water_requirement_model.pkl',
            'season_classification': 'season_classification_model.pkl'
        }
        
        for model_name, file_name in model_files.items():
            model_path = MODELS_DIR / file_name
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"✅ Loaded {model_name}")
                except Exception as e:
                    print(f"⚠️  Could not load {model_name}: {e}")
    
    def _load_encoders(self):
        """Load feature scalers and label encoders"""
        # Load scalers
        scaler_path = ENCODERS_DIR / "scalers.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)
        
        # Load label encoders
        encoder_path = ENCODERS_DIR / "label_encoders.json"
        if encoder_path.exists():
            import json
            with open(encoder_path, 'r') as f:
                self.label_encoders = json.load(f)
    
    def _load_metrics(self):
        """Load model metrics"""
        metrics_path = MODELS_DIR / "model_metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path, 'r') as f:
                self.model_metrics = json.load(f)
    
    def predict_crop_recommendation(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict recommended crop given environmental features
        
        Args:
            features: Array of shape (19,) with crop environmental features
            [temp_range, ph_range, rainfall, moisture, etc.]
        
        Returns:
            (crop_name, confidence)
        """
        if 'crop_recommendation' not in self.models:
            raise ValueError("Crop recommendation model not loaded")
        
        model = self.models['crop_recommendation']
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Get confidence
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
        
        return str(prediction), confidence
    
    def predict_crop_type(self, features: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Predict crop type (Cash, Cereal, Fruit, etc.)
        
        Args:
            features: Array of shape (26,) with engineered features
        
        Returns:
            (crop_type, probabilities_dict)
        """
        if 'crop_type_classification' not in self.models:
            raise ValueError("Crop type model not loaded")
        
        model = self.models['crop_type_classification']
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Create probability dictionary
        classes = model.classes_
        prob_dict = {
            str(cls): float(prob) 
            for cls, prob in zip(classes, probabilities)
        }
        
        return str(prediction), prob_dict
    
    def predict_growth_duration(self, features: np.ndarray) -> Tuple[float, Dict]:
        """
        Predict growth duration in days (regression)
        
        Args:
            features: Array of shape (23,) with selected features
        
        Returns:
            (predicted_days, confidence_metrics)
        """
        if 'growth_duration' not in self.models:
            raise ValueError("Growth duration model not loaded")
        
        model = self.models['growth_duration']
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict
        prediction = float(model.predict(features)[0])
        
        # Get metrics
        r2 = self.model_metrics.get('growth_duration', {}).get('r2_score', 0)
        
        return prediction, {
            'predicted_days': prediction,
            'model_r2_score': r2,
            'typical_range': self.model_metrics.get('growth_duration', {}).get('target_range', [18, 365])
        }
    
    def predict_water_requirement(self, features: np.ndarray) -> Tuple[float, Dict]:
        """
        Predict water requirement in mm/day (regression)
        
        Args:
            features: Array of shape (19,) with selected features
        
        Returns:
            (predicted_mm_per_day, metrics)
        """
        if 'water_requirement' not in self.models:
            raise ValueError("Water requirement model not loaded")
        
        model = self.models['water_requirement']
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict (scale back from [0,1])
        prediction_scaled = float(model.predict(features)[0])
        
        # Unscale using min-max range
        target_range = self.model_metrics.get('water_requirement', {}).get('target_range', [2.5, 15.0])
        prediction = prediction_scaled * (target_range[1] - target_range[0]) + target_range[0]
        
        return prediction, {
            'predicted_mm_per_day': round(prediction, 2),
            'model_r2_score': self.model_metrics.get('water_requirement', {}).get('r2_score', 0),
            'typical_range': target_range
        }
    
    def predict_season(self, features: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Predict suitable season (Kharif, Rabi, Zaid, etc.)
        
        Args:
            features: Array of shape (20,) with selected features
        
        Returns:
            (season, probabilities_dict)
        """
        if 'season_classification' not in self.models:
            raise ValueError("Season classification model not loaded")
        
        model = self.models['season_classification']
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Create probability dictionary
        classes = model.classes_
        prob_dict = {
            str(cls): float(prob) 
            for cls, prob in zip(classes, probabilities)
        }
        
        return str(prediction), prob_dict
    
    def batch_predict(self, crop_name: str, features_dict: Dict) -> Dict:
        """
        Get all predictions for a crop at once
        
        Args:
            crop_name: Name of the crop
            features_dict: Dictionary with feature arrays
        
        Returns:
            Comprehensive prediction dictionary
        """
        results = {
            'crop_name': crop_name,
            'predictions': {}
        }
        
        try:
            # Crop type prediction
            if 'crop_type_features' in features_dict:
                crop_type, type_probs = self.predict_crop_type(features_dict['crop_type_features'])
                results['predictions']['crop_type'] = {
                    'predicted': crop_type,
                    'probabilities': type_probs
                }
        except Exception as e:
            results['predictions']['crop_type'] = {'error': str(e)}
        
        try:
            # Growth duration
            if 'growth_features' in features_dict:
                days, growth_metrics = self.predict_growth_duration(features_dict['growth_features'])
                results['predictions']['growth_duration'] = growth_metrics
        except Exception as e:
            results['predictions']['growth_duration'] = {'error': str(e)}
        
        try:
            # Water requirement
            if 'water_features' in features_dict:
                water, water_metrics = self.predict_water_requirement(features_dict['water_features'])
                results['predictions']['water_requirement'] = water_metrics
        except Exception as e:
            results['predictions']['water_requirement'] = {'error': str(e)}
        
        try:
            # Season
            if 'season_features' in features_dict:
                season, season_probs = self.predict_season(features_dict['season_features'])
                results['predictions']['season'] = {
                    'predicted': season,
                    'probabilities': season_probs
                }
        except Exception as e:
            results['predictions']['season'] = {'error': str(e)}
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'models_loaded': list(self.models.keys()),
            'metrics': self.model_metrics,
            'status': 'ready' if len(self.models) > 0 else 'not_ready'
        }


# Global inference instance
_inference_instance = None


def get_inference_engine() -> ModelInference:
    """Get or create global inference engine"""
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = ModelInference()
    return _inference_instance


def make_prediction(model_type: str, features: np.ndarray, **kwargs) -> Dict:
    """Make a prediction using the inference engine"""
    engine = get_inference_engine()
    
    if model_type == 'crop_recommendation':
        crop, confidence = engine.predict_crop_recommendation(features)
        return {'crop': crop, 'confidence': confidence}
    
    elif model_type == 'crop_type':
        crop_type, probs = engine.predict_crop_type(features)
        return {'crop_type': crop_type, 'probabilities': probs}
    
    elif model_type == 'growth_duration':
        days, metrics = engine.predict_growth_duration(features)
        return metrics
    
    elif model_type == 'water_requirement':
        water, metrics = engine.predict_water_requirement(features)
        return metrics
    
    elif model_type == 'season':
        season, probs = engine.predict_season(features)
        return {'season': season, 'probabilities': probs}
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
