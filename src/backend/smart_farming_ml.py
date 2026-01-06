"""
Smart Farming ML Module
Provides crop recommendation and ML-based predictions
"""
import logging
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SmartFarmingRecommendationSystem:
    """
    ML-powered crop recommendation system using multiple models
    Supports RandomForest, GradientBoosting, and TensorFlow models
    """
    
    def __init__(self):
        """Initialize the Smart Farming system"""
        self.models_loaded = False
        self.available_models = []
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available ML models"""
        try:
            # Try to load joblib models
            import joblib
            model_dir = Path(__file__).parent / "models"
            
            # List of possible model files
            model_names = [
                "crop_recommendation_rf.joblib",
                "crop_recommendation_gb.joblib",
                "yield_prediction_model.joblib",
                "water_model.joblib",
            ]
            
            for model_name in model_names:
                model_path = model_dir / model_name
                if model_path.exists():
                    try:
                        model = joblib.load(model_path)
                        self.available_models.append({
                            'name': model_name,
                            'model': model,
                            'type': 'joblib'
                        })
                        logger.info(f"Loaded model: {model_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {e}")
            
            if self.available_models:
                self.models_loaded = True
                logger.info(f"Loaded {len(self.available_models)} ML models")
            else:
                logger.warning("No ML models found, using fallback recommendations")
        
        except ImportError:
            logger.warning("joblib not available, using fallback recommendations")
    
    def recommend_crop(self, 
                      temperature: float,
                      humidity: float,
                      ph: float,
                      rainfall: float,
                      nitrogen: float,
                      phosphorus: float,
                      potassium: float,
                      soil_type: str = "loamy") -> Dict[str, Any]:
        """
        Recommend crops based on environmental and soil conditions
        
        Args:
            temperature: Average temperature in Celsius
            humidity: Humidity percentage (0-100)
            ph: Soil pH (0-14)
            rainfall: Annual rainfall in mm
            nitrogen: Soil nitrogen in kg/ha
            phosphorus: Soil phosphorus in kg/ha
            potassium: Soil potassium in kg/ha
            soil_type: Type of soil (loamy, clay, sandy)
        
        Returns:
            Dictionary with crop recommendations
        """
        # Prepare features for model
        features = {
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall,
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
        }
        
        # If models are loaded, use them
        if self.models_loaded and self.available_models:
            return self._predict_with_models(features)
        
        # Fallback to rule-based recommendations
        return self._rule_based_recommendation(features)
    
    def _predict_with_models(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Use loaded ML models for prediction"""
        try:
            # Simple prediction using first available model
            model_info = self.available_models[0]
            model = model_info['model']
            
            # Prepare features in the order the model expects
            feature_list = [
                features.get('temperature', 25),
                features.get('humidity', 60),
                features.get('ph', 6.5),
                features.get('rainfall', 1000),
                features.get('nitrogen', 40),
                features.get('phosphorus', 20),
                features.get('potassium', 15),
            ]
            
            # Make prediction
            if hasattr(model, 'predict'):
                prediction = model.predict([feature_list])
                return {
                    'success': True,
                    'recommended_crop': prediction[0] if isinstance(prediction, (list, tuple)) else str(prediction),
                    'model_used': model_info['name'],
                    'confidence': 'Medium (ML Model)',
                }
        except Exception as e:
            logger.error(f"ML prediction failed: {e}, falling back to rules")
        
        return self._rule_based_recommendation(features)
    
    def _rule_based_recommendation(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Rule-based fallback for crop recommendation"""
        temp = features.get('temperature', 25)
        humidity = features.get('humidity', 60)
        rainfall = features.get('rainfall', 1000)
        ph = features.get('ph', 6.5)
        
        # Simple rule-based logic
        recommendations = []
        
        # Wheat prefers cool, dry climate
        if 15 <= temp <= 25 and rainfall < 500:
            recommendations.append(('Wheat', 0.9))
        
        # Rice needs warm and wet
        if 20 <= temp <= 30 and humidity > 70 and rainfall > 1200:
            recommendations.append(('Rice', 0.9))
        
        # Corn (Maize) - moderate conditions
        if 18 <= temp <= 27 and humidity > 50:
            recommendations.append(('Maize', 0.85))
        
        # Cotton - warm, moderate rain
        if 21 <= temp <= 30 and 400 < rainfall < 1000:
            recommendations.append(('Cotton', 0.8))
        
        # Sugarcane - warm, high rainfall
        if 21 <= temp <= 27 and rainfall > 1000:
            recommendations.append(('Sugarcane', 0.85))
        
        # Vegetables - varied conditions
        if 15 <= temp <= 25:
            recommendations.append(('Vegetables', 0.75))
        
        if recommendations:
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return {
                'success': True,
                'recommended_crop': recommendations[0][0],
                'confidence': f'{recommendations[0][1]:.0%} (Rule-based)',
                'alternatives': [crop for crop, _ in recommendations[1:3]],
            }
        
        return {
            'success': True,
            'recommended_crop': 'Vegetables',
            'confidence': '50% (Generic)',
            'note': 'Conditions not ideal for major crops, consider vegetables'
        }
    
    def prepare_models(self):
        """Prepare/train models (placeholder for actual training)"""
        logger.info("Models prepared (using pre-trained weights)")
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'models_loaded': self.models_loaded,
            'count': len(self.available_models),
            'available_models': [m['name'] for m in self.available_models],
            'fallback_mode': not self.models_loaded,
        }


# Create a singleton instance
_system = None

def get_smart_farming_system() -> SmartFarmingRecommendationSystem:
    """Get or create the smart farming system instance"""
    global _system
    if _system is None:
        _system = SmartFarmingRecommendationSystem()
    return _system
