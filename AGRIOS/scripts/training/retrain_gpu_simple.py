#!/usr/bin/env python3
"""
Simplified GPU-Accelerated Model Retraining for AgriSense
Uses available datasets and synthetic data with TensorFlow GPU
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any
import joblib

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUAcceleratedModelTrainer:
    """Retrain ML models with GPU acceleration"""
    
    def __init__(self, backend_path: Path = Path('agrisense_app/backend')):
        self.backend_path = Path(backend_path)
        self.models_dir = self.backend_path / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify GPU
        self.gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"GPUs available: {len(self.gpus)}")
        
        if self.gpus:
            # Enable memory growth
            for gpu in self.gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("✅ GPU memory growth enabled")
    
    def load_crop_recommendation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare crop recommendation dataset"""
        logger.info("📊 Loading Crop Recommendation Dataset...")
        
        csv_path = self.backend_path / 'Crop_recommendation.csv'
        if not csv_path.exists():
            logger.warning(f"  ⚠️ {csv_path} not found, generating synthetic data...")
            return self._generate_synthetic_crop_data()
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"  ✓ Loaded {len(df)} samples, {len(df.columns)} features")
            
            # Assuming last column is target (label)
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values
            
            # Normalize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            return X, y
        except Exception as e:
            logger.warning(f"  ⚠️ Error loading data: {e}, using synthetic data...")
            return self._generate_synthetic_crop_data()
    
    def _generate_synthetic_crop_data(self, n_samples: int = 2200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic crop/disease data"""
        logger.info(f"  📈 Generating {n_samples} synthetic samples...")
        
        # Features: N, P, K, temperature, humidity, ph, rainfall
        X = np.random.rand(n_samples, 7).astype(np.float32) * 100
        
        # Labels: Crop types or disease types
        crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Groundnut', 'Pulses']
        y = np.random.choice(crops, n_samples)
        
        return X, y
    
    def train_gradient_boosting_model(self, X: np.ndarray, y: np.ndarray, model_name: str = 'crop_model') -> Dict[str, Any]:
        """Train GradientBoosting model (CPU-optimized)"""
        logger.info(f"\n🎯 Training GradientBoosting {model_name}...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_iter_no_change=10,
            validation_fraction=0.1
        )
        
        logger.info(f"  Training on {len(X_train)} samples...")
        model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        logger.info(f"  ✅ Train Accuracy: {train_acc:.4f}")
        logger.info(f"  ✅ Test Accuracy: {test_acc:.4f}")
        
        # Save model
        model_path = self.models_dir / f'{model_name}_gb.joblib'
        joblib.dump(model, model_path)
        logger.info(f"  💾 Model saved to {model_path}")
        
        return {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'model_type': 'GradientBoosting',
            'path': str(model_path)
        }
    
    def train_neural_network(self, X: np.ndarray, y: np.ndarray, model_name: str = 'crop_model') -> Dict[str, Any]:
        """Train neural network with GPU acceleration (using CPU device context for compatibility)"""
        logger.info(f"\n🧠 Training Neural Network {model_name} (Keras)...")
        
        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        n_classes = len(encoder.classes_)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Use CPU for model building (to avoid CUDA compatibility issues)
        # Training will still use GPU if available via Keras backend
        with tf.device('/CPU:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(n_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        logger.info(f"  Training on GPU..." if self.gpus else "  Training on CPU...")
        
        history = model.fit(
            X_train, y_train,
            epochs=30,  # Reduced epochs for faster training
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"  ✅ Train Accuracy: {train_acc:.4f}")
        logger.info(f"  ✅ Test Accuracy: {test_acc:.4f}")
        
        # Save model
        model_path = self.models_dir / f'{model_name}_nn.keras'
        model.save(model_path)
        logger.info(f"  💾 Model saved to {model_path}")
        
        return {
            'model': model,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'model_type': 'NeuralNetwork',
            'path': str(model_path),
            'classes': encoder.classes_.tolist()
        }
    
    def retrain_all(self) -> Dict[str, Any]:
        """Retrain all models"""
        logger.info("\n" + "="*70)
        logger.info("🚀 GPU-ACCELERATED MODEL RETRAINING")
        logger.info("="*70)
        
        results = {}
        
        # Load data
        X, y = self.load_crop_recommendation_data()
        
        # Train crop recommendation models
        logger.info("\n📌 CROP RECOMMENDATION MODELS")
        results['crop_gb'] = self.train_gradient_boosting_model(X, y, 'crop_recommendation')
        results['crop_nn'] = self.train_neural_network(X, y, 'crop_recommendation')
        
        # Train disease detection models (synthetic data)
        logger.info("\n📌 DISEASE DETECTION MODELS")
        X_disease, y_disease = self._generate_synthetic_crop_data(1500)
        results['disease_gb'] = self.train_gradient_boosting_model(X_disease, y_disease, 'disease_detection')
        results['disease_nn'] = self.train_neural_network(X_disease, y_disease, 'disease_detection')
        
        # Train weed management models (synthetic data)
        logger.info("\n📌 WEED MANAGEMENT MODELS")
        X_weed, y_weed = self._generate_synthetic_crop_data(1200)
        results['weed_gb'] = self.train_gradient_boosting_model(X_weed, y_weed, 'weed_management')
        results['weed_nn'] = self.train_neural_network(X_weed, y_weed, 'weed_management')
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("📊 RETRAINING SUMMARY")
        logger.info("="*70)
        
        for model_name, result in results.items():
            logger.info(f"✅ {model_name}:")
            logger.info(f"   Type: {result['model_type']}")
            logger.info(f"   Test Accuracy: {result['test_accuracy']:.4f}")
            logger.info(f"   Saved: {result['path']}")
        
        logger.info("\n🎯 All models retrained and saved!")
        logger.info("Models directory:", str(self.models_dir))
        
        return results

def main():
    trainer = GPUAcceleratedModelTrainer()
    results = trainer.retrain_all()
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'gpu_available': len(trainer.gpus) > 0,
        'tensorflow_version': tf.__version__,
        'models_trained': list(results.keys()),
        'summary': {k: {
            'type': v.get('model_type'),
            'test_accuracy': v.get('test_accuracy'),
            'path': v.get('path')
        } for k, v in results.items()}
    }
    
    report_path = trainer.models_dir / f'retraining_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n📄 Report saved to: {report_path}")

if __name__ == '__main__':
    main()
