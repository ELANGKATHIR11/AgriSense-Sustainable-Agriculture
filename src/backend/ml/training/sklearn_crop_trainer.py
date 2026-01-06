#!/usr/bin/env python3
"""
AgriSense Scikit-Learn Crop Recommendation Trainer
===================================================
Alternative trainer using RandomForest when CatBoost is unavailable.

Features:
- RandomForestClassifier with SMOTE
- Feature importance analysis
- Cross-validation
- ONNX export via skl2onnx

Usage:
    python sklearn_crop_trainer.py --csv ../data/tabular/india_crops_complete.csv

Author: AgriSense ML Team
"""

import os
import sys
import json
import logging
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try optional imports
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning("imbalanced-learn not installed. SMOTE disabled.")

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.info("skl2onnx not installed. ONNX export disabled.")


class SklearnCropTrainer:
    """
    Scikit-learn based crop recommendation trainer.
    
    Features:
    - RandomForest or GradientBoosting
    - SMOTE for class imbalance
    - Cross-validation
    - Feature importance
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.class_names = []
        self.metrics = {}
        
    @staticmethod
    def _default_config() -> Dict:
        return {
            'model_type': 'random_forest',  # 'random_forest' or 'gradient_boosting'
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'n_jobs': -1,
            'random_state': 42,
            'use_smote': True,
            'cv_folds': 5,
            'output_dir': './models/tabular'
        }
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and 'label' column
            
        Returns:
            X, y arrays
        """
        logger.info("Preparing data...")
        
        # Identify columns
        target_col = 'label'
        if target_col not in df.columns:
            # Try to find target column
            possible_targets = ['crop', 'Crop', 'Label', 'class']
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    break
        
        if target_col not in df.columns:
            raise ValueError(f"Target column not found. Available: {df.columns.tolist()}")
        
        # Feature columns (numeric only for simplicity)
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        if not feature_cols:
            # Use all numeric columns except target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)
        
        self.feature_names = feature_cols
        logger.info(f"Features: {feature_cols}")
        
        # Extract X and y
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_.tolist()
        
        logger.info(f"X shape: {X.shape}, Classes: {len(self.class_names)}")
        
        return X, y_encoded
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for class imbalance."""
        if not IMBLEARN_AVAILABLE or not self.config['use_smote']:
            return X, y
        
        logger.info("Applying SMOTE...")
        smote = SMOTE(random_state=self.config['random_state'])
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"After SMOTE: {X_resampled.shape[0]} samples")
        
        return X_resampled, y_resampled
    
    def build_model(self):
        """Build the model."""
        if self.config['model_type'] == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                min_samples_split=self.config['min_samples_split'],
                min_samples_leaf=self.config['min_samples_leaf'],
                n_jobs=self.config['n_jobs'],
                random_state=self.config['random_state'],
                class_weight='balanced'
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                min_samples_split=self.config['min_samples_split'],
                min_samples_leaf=self.config['min_samples_leaf'],
                random_state=self.config['random_state']
            )
        
        logger.info(f"Built {self.config['model_type']} model")
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the model.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dict with metrics
        """
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Apply SMOTE on training data only
        X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)
        
        # Build model
        self.build_model()
        
        # Cross-validation
        logger.info(f"Running {self.config['cv_folds']}-fold cross-validation...")
        cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, 
                            random_state=self.config['random_state'])
        cv_scores = cross_val_score(self.model, X_train_resampled, y_train_resampled, 
                                    cv=cv, scoring='accuracy')
        
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Train final model
        logger.info("Training final model...")
        self.model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'n_classes': len(self.class_names),
            'n_samples': len(df),
            'n_features': len(self.feature_names)
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            self.metrics['feature_importance'] = {
                name: float(imp) 
                for name, imp in zip(self.feature_names, importance)
            }
        
        logger.info(f"\n{'='*60}")
        logger.info("Training Results:")
        logger.info(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score: {self.metrics['f1_macro']:.4f}")
        logger.info(f"  CV Score: {self.metrics['cv_accuracy_mean']:.4f}")
        
        # Print classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=False
        )
        logger.info(f"\nClassification Report:\n{report}")
        
        return self.metrics
    
    def save(self, output_dir: str = None):
        """Save model and artifacts."""
        output_dir = Path(output_dir or self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / 'crop_recommendation_rf.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'config': self.config
            }, f)
        logger.info(f"✓ Saved model: {model_path}")
        
        # Save metrics
        metrics_path = output_dir / 'training_metrics.json'
        # Convert numpy types for JSON
        metrics_json = {}
        for k, v in self.metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                metrics_json[k] = float(v)
            elif isinstance(v, dict):
                metrics_json[k] = {kk: float(vv) for kk, vv in v.items()}
            else:
                metrics_json[k] = v
        
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': metrics_json,
                'timestamp': datetime.now().isoformat(),
                'model_type': self.config['model_type']
            }, f, indent=2)
        logger.info(f"✓ Saved metrics: {metrics_path}")
        
        # Export to ONNX
        if ONNX_AVAILABLE:
            try:
                onnx_path = output_dir / 'crop_recommendation.onnx'
                initial_type = [('input', FloatTensorType([None, len(self.feature_names)]))]
                onnx_model = convert_sklearn(self.model, initial_types=initial_type)
                with open(onnx_path, 'wb') as f:
                    f.write(onnx_model.SerializeToString())
                logger.info(f"✓ Exported ONNX: {onnx_path}")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")
        
        return str(model_path)
    
    @classmethod
    def load(cls, model_path: str) -> 'SklearnCropTrainer':
        """Load model from file."""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        trainer = cls(config=data['config'])
        trainer.model = data['model']
        trainer.scaler = data['scaler']
        trainer.label_encoder = data['label_encoder']
        trainer.feature_names = data['feature_names']
        trainer.class_names = data['class_names']
        
        logger.info(f"✓ Loaded model from {model_path}")
        return trainer
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            predictions, probabilities
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode labels
        labels = self.label_encoder.inverse_transform(predictions)
        
        return labels, probabilities


def main():
    parser = argparse.ArgumentParser(description='Train sklearn crop recommendation model')
    parser.add_argument('--csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--output-dir', type=str, default='./models/tabular')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting'])
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=15)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--no-smote', action='store_true')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🌾 AgriSense Sklearn Crop Trainer")
    print("=" * 70)
    
    # Load data
    logger.info(f"Loading data from: {args.csv}")
    df = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df)} samples")
    
    # Configure
    config = {
        'model_type': args.model_type,
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'n_jobs': -1,
        'random_state': 42,
        'use_smote': not args.no_smote,
        'cv_folds': args.cv_folds,
        'output_dir': args.output_dir
    }
    
    # Train
    trainer = SklearnCropTrainer(config)
    metrics = trainer.train(df)
    
    # Save
    trainer.save(args.output_dir)
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1_macro']:.4f}")
    print(f"  CV Score:  {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']*2:.4f})")
    print(f"\nModel saved to: {args.output_dir}")
    
    # Feature importance
    if 'feature_importance' in metrics:
        print("\nFeature Importance:")
        sorted_features = sorted(
            metrics['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for name, imp in sorted_features:
            print(f"  {name}: {imp:.4f}")


if __name__ == '__main__':
    main()
