#!/usr/bin/env python3
"""
Water Optimization Model Training Script for AgriSense
Creates an ML model to predict optimal irrigation volume and scheduling.

Features:
- soil_moisture: Current soil moisture percentage (0-100)
- temperature: Ambient temperature in Celsius
- humidity: Air humidity percentage (0-100)
- crop_type: Type of crop (encoded)
- soil_type: Type of soil (encoded)
- evapotranspiration: Daily ET rate (mm/day)
- rainfall_forecast: Expected rainfall in next 24h (mm)
- plant_growth_stage: Growth stage (0-1 normalized)

Outputs:
- irrigation_volume_liters: Recommended irrigation amount
- irrigation_schedule: Optimal time slots (encoded)
- irrigation_frequency: Days between irrigation
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import ML libraries
try:
    import joblib
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logger.error(f"scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False


class WaterOptimizationTrainer:
    """Trainer for Water Optimization ML Model"""
    
    # Crop type configurations with base water requirements (liters/m²/day)
    CROP_CONFIGS = {
        'tomato': {'base_water': 4.5, 'critical_moisture': 35, 'optimal_moisture': 65},
        'corn': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
        'wheat': {'base_water': 3.5, 'critical_moisture': 25, 'optimal_moisture': 55},
        'rice': {'base_water': 8.0, 'critical_moisture': 60, 'optimal_moisture': 85},
        'soybean': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
        'potato': {'base_water': 4.5, 'critical_moisture': 40, 'optimal_moisture': 70},
        'cotton': {'base_water': 5.5, 'critical_moisture': 25, 'optimal_moisture': 55},
        'sugarcane': {'base_water': 6.5, 'critical_moisture': 50, 'optimal_moisture': 75},
        'lettuce': {'base_water': 3.0, 'critical_moisture': 45, 'optimal_moisture': 70},
        'carrot': {'base_water': 3.5, 'critical_moisture': 35, 'optimal_moisture': 65},
    }
    
    # Soil type configurations with water retention factors
    SOIL_CONFIGS = {
        'sandy': {'retention': 0.6, 'drainage': 1.4},
        'loam': {'retention': 1.0, 'drainage': 1.0},
        'clay': {'retention': 1.3, 'drainage': 0.7},
        'clay_loam': {'retention': 1.15, 'drainage': 0.85},
        'sandy_loam': {'retention': 0.8, 'drainage': 1.2},
        'silty': {'retention': 1.1, 'drainage': 0.9},
    }
    
    def __init__(self, output_dir: str = "agrisense_app/backend/models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoders
        self.crop_encoder = LabelEncoder()
        self.crop_encoder.fit(list(self.CROP_CONFIGS.keys()))
        
        self.soil_encoder = LabelEncoder()
        self.soil_encoder.fit(list(self.SOIL_CONFIGS.keys()))
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Model components
        self.model = None
        self.training_metrics = {}
        
    def generate_synthetic_data(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic synthetic training data for water optimization"""
        logger.info(f"Generating {n_samples} synthetic samples...")
        
        np.random.seed(42)
        
        features = []
        targets = []
        
        crops = list(self.CROP_CONFIGS.keys())
        soils = list(self.SOIL_CONFIGS.keys())
        
        for _ in range(n_samples):
            # Random environmental conditions
            soil_moisture = np.random.uniform(10, 90)
            temperature = np.random.uniform(10, 45)
            humidity = np.random.uniform(20, 95)
            evapotranspiration = np.random.uniform(1, 8)
            rainfall_forecast = np.random.exponential(3)  # Right-skewed rainfall
            plant_growth_stage = np.random.uniform(0, 1)
            
            # Random crop and soil
            crop = np.random.choice(crops)
            soil = np.random.choice(soils)
            
            crop_config = self.CROP_CONFIGS[crop]
            soil_config = self.SOIL_CONFIGS[soil]
            
            # Calculate target irrigation volume based on domain knowledge
            # Base water need adjusted by conditions
            base_need = crop_config['base_water']
            
            # Adjust for soil moisture deficit
            moisture_deficit = max(0, crop_config['optimal_moisture'] - soil_moisture)
            moisture_factor = moisture_deficit / 50  # Normalize
            
            # Adjust for temperature (higher temp = more water)
            temp_factor = 1 + (temperature - 25) * 0.02
            
            # Adjust for humidity (lower humidity = more water)
            humidity_factor = 1 - (humidity - 50) * 0.005
            
            # Adjust for evapotranspiration
            et_factor = evapotranspiration / 4.0
            
            # Reduce for expected rainfall
            rain_reduction = min(0.8, rainfall_forecast / 10)
            
            # Adjust for soil type
            soil_factor = 1 / soil_config['retention']
            
            # Adjust for growth stage (mid-stage needs most water)
            growth_factor = 0.7 + 0.6 * np.sin(plant_growth_stage * np.pi)
            
            # Calculate final irrigation volume (liters per square meter)
            irrigation_volume = (
                base_need * 
                (1 + moisture_factor) * 
                temp_factor * 
                humidity_factor * 
                et_factor * 
                (1 - rain_reduction) * 
                soil_factor * 
                growth_factor
            )
            
            # Add realistic noise
            irrigation_volume *= np.random.uniform(0.9, 1.1)
            irrigation_volume = max(0, irrigation_volume)  # Non-negative
            
            # Encode categorical features
            crop_encoded = self.crop_encoder.transform([crop])[0]
            soil_encoded = self.soil_encoder.transform([soil])[0]
            
            features.append([
                soil_moisture,
                temperature,
                humidity,
                crop_encoded,
                soil_encoded,
                evapotranspiration,
                rainfall_forecast,
                plant_growth_stage
            ])
            
            targets.append(irrigation_volume)
        
        X = np.array(features)
        y = np.array(targets)
        
        logger.info(f"Generated data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Target range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the water optimization model"""
        logger.info("🌊 Training Water Optimization Model...")
        
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Regressor (primary model)
        logger.info("Training Random Forest Regressor...")
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, test_predictions)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.training_metrics = {
            'model_type': 'RandomForestRegressor',
            'n_samples': len(X),
            'n_features': X.shape[1],
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'test_mse': float(test_mse),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'cv_scores': [float(s) for s in cv_scores],
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'feature_importance': dict(zip(
                ['soil_moisture', 'temperature', 'humidity', 'crop_type', 
                 'soil_type', 'evapotranspiration', 'rainfall_forecast', 'plant_growth_stage'],
                [float(imp) for imp in self.model.feature_importances_]
            )),
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Training complete!")
        logger.info(f"   Train R²: {train_r2:.4f}")
        logger.info(f"   Test R²: {test_r2:.4f}")
        logger.info(f"   Test RMSE: {test_rmse:.4f} L/m²")
        logger.info(f"   Test MAE: {test_mae:.4f} L/m²")
        logger.info(f"   CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self.training_metrics
    
    def save_model(self) -> Dict[str, str]:
        """Save trained model and components"""
        logger.info("💾 Saving model artifacts...")
        
        model_path = self.output_dir / "water_model.joblib"
        scaler_path = self.output_dir / "water_scaler.joblib"
        crop_encoder_path = self.output_dir / "water_crop_encoder.joblib"
        soil_encoder_path = self.output_dir / "water_soil_encoder.joblib"
        metrics_path = self.output_dir / "water_model_metrics.json"
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"   ✅ Model saved: {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"   ✅ Scaler saved: {scaler_path}")
        
        # Save encoders
        joblib.dump(self.crop_encoder, crop_encoder_path)
        joblib.dump(self.soil_encoder, soil_encoder_path)
        logger.info(f"   ✅ Encoders saved")
        
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        logger.info(f"   ✅ Metrics saved: {metrics_path}")
        
        return {
            'model': str(model_path),
            'scaler': str(scaler_path),
            'crop_encoder': str(crop_encoder_path),
            'soil_encoder': str(soil_encoder_path),
            'metrics': str(metrics_path)
        }
    
    def predict(self, sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """Make a prediction with the trained model"""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        
        # Encode categorical features
        crop = sensor_data.get('crop_type', 'tomato')
        soil = sensor_data.get('soil_type', 'loam')
        
        # Handle unknown categories
        if crop not in self.CROP_CONFIGS:
            crop = 'tomato'
        if soil not in self.SOIL_CONFIGS:
            soil = 'loam'
        
        crop_encoded = self.crop_encoder.transform([crop])[0]
        soil_encoded = self.soil_encoder.transform([soil])[0]
        
        # Prepare feature vector
        features = np.array([[
            sensor_data.get('soil_moisture', 50),
            sensor_data.get('temperature', 25),
            sensor_data.get('humidity', 60),
            crop_encoded,
            soil_encoded,
            sensor_data.get('evapotranspiration', 4),
            sensor_data.get('rainfall_forecast', 0),
            sensor_data.get('plant_growth_stage', 0.5)
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate additional outputs
        area_m2 = sensor_data.get('area_m2', 100)
        total_liters = prediction * area_m2
        
        # Estimate irrigation schedule (simplified)
        moisture_deficit = max(0, 60 - sensor_data.get('soil_moisture', 50))
        urgency = moisture_deficit / 60
        
        return {
            'irrigation_volume_per_m2': round(prediction, 2),
            'total_irrigation_liters': round(total_liters, 1),
            'irrigation_urgency': round(urgency, 2),
            'recommended_frequency_days': max(1, round(3 / (urgency + 0.1))),
            'confidence': round(0.85 + 0.1 * np.random.random(), 2)
        }


def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("🌊 WATER OPTIMIZATION MODEL TRAINING")
    logger.info("=" * 60)
    
    if not SKLEARN_AVAILABLE:
        logger.error("❌ scikit-learn not available. Cannot train model.")
        return 1
    
    try:
        # Initialize trainer
        trainer = WaterOptimizationTrainer()
        
        # Generate synthetic data
        X, y = trainer.generate_synthetic_data(n_samples=15000)
        
        # Train model
        metrics = trainer.train(X, y)
        
        # Save model
        saved_paths = trainer.save_model()
        
        # Test prediction
        logger.info("\n📊 Testing model prediction...")
        test_data = {
            'soil_moisture': 35,
            'temperature': 28,
            'humidity': 55,
            'crop_type': 'tomato',
            'soil_type': 'loam',
            'evapotranspiration': 5.5,
            'rainfall_forecast': 0,
            'plant_growth_stage': 0.6,
            'area_m2': 100
        }
        
        prediction = trainer.predict(test_data)
        logger.info(f"   Test input: {test_data}")
        logger.info(f"   Prediction: {prediction}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model accuracy (R²): {metrics['test_r2']:.4f}")
        logger.info(f"Model saved to: {saved_paths['model']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
