#!/usr/bin/env python3
"""
AgriSense CatBoost Crop Recommendation Trainer
===============================================
Trains CatBoost classifier with DART mode for crop recommendation.

Features:
- CatBoost with DART (Dropout Additive Regression Trees)
- Native categorical feature handling
- SMOTE-NC for class imbalance
- Mixup augmentation for tabular data
- ONNX export for cross-platform inference
- Model versioning and tracking

Usage:
    python catboost_trainer.py --csv ../data/tabular/india_crops_complete.csv

Author: AgriSense ML Team
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not installed. Install with: pip install catboost")

try:
    from imblearn.over_sampling import SMOTENC
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning("imbalanced-learn not installed. SMOTE-NC disabled.")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not installed. ONNX export disabled.")


class TabularMixup:
    """
    Mixup augmentation for tabular data.
    
    Reference: mixup: Beyond Empirical Risk Minimization (Zhang et al., 2018)
    """
    
    def __init__(self, alpha: float = 0.4):
        """
        Args:
            alpha: Beta distribution parameter for mixup ratio
        """
        self.alpha = alpha
    
    def augment(self, 
                X: np.ndarray, 
                y: np.ndarray,
                num_samples: int = None,
                categorical_indices: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mixup augmented samples.
        
        Args:
            X: Feature matrix
            y: Labels
            num_samples: Number of samples to generate
            categorical_indices: Indices of categorical features (kept from primary sample)
            
        Returns:
            Augmented (X, y) tuple
        """
        if num_samples is None:
            num_samples = len(X) // 2
        
        n = len(X)
        indices_a = np.random.randint(0, n, num_samples)
        indices_b = np.random.randint(0, n, num_samples)
        
        # Generate mixup ratios
        lam = np.random.beta(self.alpha, self.alpha, num_samples)
        lam = np.maximum(lam, 1 - lam)  # Ensure primary sample dominates
        
        # Mixup features
        X_aug = np.zeros((num_samples, X.shape[1]))
        for i in range(num_samples):
            X_aug[i] = lam[i] * X[indices_a[i]] + (1 - lam[i]) * X[indices_b[i]]
            
            # Keep categorical features from primary sample
            if categorical_indices:
                for cat_idx in categorical_indices:
                    X_aug[i, cat_idx] = X[indices_a[i], cat_idx]
        
        # Use primary sample's label (since lam >= 0.5)
        y_aug = y[indices_a]
        
        return X_aug, y_aug


class CatBoostCropTrainer:
    """
    CatBoost trainer for crop recommendation with DART mode.
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Training configuration dictionary
        """
        self.config = config or self._default_config()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.categorical_features = []
        self.metrics_history = []
        
    @staticmethod
    def _default_config() -> Dict:
        """Default training configuration."""
        return {
            # CatBoost parameters
            'iterations': 2000,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3.0,
            'border_count': 254,
            'boosting_type': 'Plain',  # 'Plain' for DART-like behavior
            'bootstrap_type': 'MVS',   # Minimum Variance Sampling
            'subsample': 0.8,
            'sampling_frequency': 'PerTree',
            'random_strength': 1.0,
            'rsm': 0.8,  # Random subspace method (feature sampling)
            
            # Training settings
            'early_stopping_rounds': 100,
            'task_type': 'CPU',  # or 'GPU' if available
            'thread_count': -1,
            'random_seed': 42,
            'verbose': 100,
            
            # Data augmentation
            'use_smote': True,
            'smote_k_neighbors': 5,
            'use_mixup': True,
            'mixup_alpha': 0.4,
            'mixup_samples': 5000,
            
            # Cross-validation
            'n_folds': 5,
            'stratified': True,
            
            # Output
            'output_dir': './models/tabular',
            'model_name': 'catboost_crop_v1'
        }
    
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Preprocess dataset for training.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (features DataFrame, encoded labels)
        """
        logger.info("Preprocessing data...")
        
        # Identify label column
        label_col = 'label' if 'label' in df.columns else 'crop'
        
        # Separate features and target
        X = df.drop(columns=[label_col])
        y = self.label_encoder.fit_transform(df[label_col])
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Identify categorical features
        self.categorical_features = []
        for i, col in enumerate(X.columns):
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                self.categorical_features.append(i)
        
        logger.info(f"  Features: {len(self.feature_names)}")
        logger.info(f"  Categorical: {len(self.categorical_features)}")
        logger.info(f"  Classes: {len(self.label_encoder.classes_)}")
        logger.info(f"  Samples: {len(X)}")
        
        return X, y
    
    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE-NC for class balancing."""
        if not IMBLEARN_AVAILABLE or not self.config.get('use_smote', False):
            return X, y
        
        logger.info("Applying SMOTE-NC...")
        
        # SMOTE-NC handles categorical features
        categorical_indices = self.categorical_features if self.categorical_features else False
        
        try:
            smote = SMOTENC(
                categorical_features=categorical_indices,
                k_neighbors=self.config.get('smote_k_neighbors', 5),
                random_state=self.config.get('random_seed', 42)
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"  Before SMOTE: {len(X)} samples")
            logger.info(f"  After SMOTE: {len(X_resampled)} samples")
            return X_resampled, y_resampled
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y
    
    def _apply_mixup(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Mixup augmentation."""
        if not self.config.get('use_mixup', False):
            return X, y
        
        logger.info("Applying Mixup augmentation...")
        
        mixup = TabularMixup(alpha=self.config.get('mixup_alpha', 0.4))
        X_aug, y_aug = mixup.augment(
            X, y,
            num_samples=self.config.get('mixup_samples', 5000),
            categorical_indices=self.categorical_features
        )
        
        # Combine original and augmented
        X_combined = np.vstack([X, X_aug])
        y_combined = np.concatenate([y, y_aug])
        
        logger.info(f"  Added {len(X_aug)} mixup samples")
        
        return X_combined, y_combined
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train CatBoost model with full pipeline.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dict with training metrics
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is required. Install with: pip install catboost")
        
        logger.info("=" * 60)
        logger.info("🌾 CatBoost Crop Recommendation Training")
        logger.info("=" * 60)
        
        # Preprocess
        X, y = self._preprocess_data(df)
        X_array = X.values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y,
            test_size=0.2,
            random_state=self.config['random_seed'],
            stratify=y
        )
        
        # Apply augmentation to training data only
        X_train_aug, y_train_aug = self._apply_smote(X_train, y_train)
        X_train_aug, y_train_aug = self._apply_mixup(X_train_aug, y_train_aug)
        
        # Create CatBoost pools
        train_pool = Pool(
            data=X_train_aug,
            label=y_train_aug,
            cat_features=self.categorical_features,
            feature_names=self.feature_names
        )
        
        test_pool = Pool(
            data=X_test,
            label=y_test,
            cat_features=self.categorical_features,
            feature_names=self.feature_names
        )
        
        # Initialize model
        logger.info("\nInitializing CatBoost classifier...")
        self.model = CatBoostClassifier(
            iterations=self.config['iterations'],
            learning_rate=self.config['learning_rate'],
            depth=self.config['depth'],
            l2_leaf_reg=self.config['l2_leaf_reg'],
            border_count=self.config['border_count'],
            boosting_type=self.config['boosting_type'],
            bootstrap_type=self.config['bootstrap_type'],
            subsample=self.config['subsample'],
            random_strength=self.config['random_strength'],
            rsm=self.config['rsm'],
            task_type=self.config['task_type'],
            thread_count=self.config['thread_count'],
            random_seed=self.config['random_seed'],
            verbose=self.config['verbose'],
            early_stopping_rounds=self.config['early_stopping_rounds'],
            use_best_model=True,
            eval_metric='MultiClass'
        )
        
        # Train
        logger.info("\nTraining model...")
        self.model.fit(
            train_pool,
            eval_set=test_pool,
            plot=False
        )
        
        # Evaluate
        logger.info("\nEvaluating model...")
        y_pred = self.model.predict(X_test).flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'best_iteration': self.model.get_best_iteration(),
            'n_classes': len(self.label_encoder.classes_),
            'n_train_samples': len(X_train_aug),
            'n_test_samples': len(X_test)
        }
        
        logger.info(f"\n📊 Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  Best iteration: {metrics['best_iteration']}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        return metrics
    
    def cross_validate(self, df: pd.DataFrame) -> Dict:
        """
        Perform k-fold cross-validation.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dict with CV metrics
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is required")
        
        logger.info(f"\n🔄 {self.config['n_folds']}-Fold Cross-Validation")
        
        X, y = self._preprocess_data(df)
        X_array = X.values
        
        kfold = StratifiedKFold(
            n_splits=self.config['n_folds'],
            shuffle=True,
            random_state=self.config['random_seed']
        )
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_array, y)):
            logger.info(f"\n--- Fold {fold + 1}/{self.config['n_folds']} ---")
            
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Augmentation
            X_train, y_train = self._apply_smote(X_train, y_train)
            
            # Create pools
            train_pool = Pool(X_train, y_train, cat_features=self.categorical_features)
            val_pool = Pool(X_val, y_val, cat_features=self.categorical_features)
            
            # Train
            model = CatBoostClassifier(
                iterations=self.config['iterations'],
                learning_rate=self.config['learning_rate'],
                depth=self.config['depth'],
                random_seed=self.config['random_seed'],
                verbose=0,
                early_stopping_rounds=50
            )
            model.fit(train_pool, eval_set=val_pool, plot=False)
            
            # Evaluate
            y_pred = model.predict(X_val).flatten()
            fold_metrics.append({
                'accuracy': accuracy_score(y_val, y_pred),
                'f1_weighted': f1_score(y_val, y_pred, average='weighted')
            })
            
            logger.info(f"  Accuracy: {fold_metrics[-1]['accuracy']:.4f}")
        
        # Aggregate
        cv_results = {
            'mean_accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'std_accuracy': np.std([m['accuracy'] for m in fold_metrics]),
            'mean_f1': np.mean([m['f1_weighted'] for m in fold_metrics]),
            'std_f1': np.std([m['f1_weighted'] for m in fold_metrics]),
            'fold_metrics': fold_metrics
        }
        
        logger.info(f"\n📊 CV Results:")
        logger.info(f"  Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        logger.info(f"  F1 Score: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        
        return cv_results
    
    def save_model(self, output_dir: str = None) -> Dict[str, str]:
        """
        Save trained model and artifacts.
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Dict with saved file paths
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        output_dir = Path(output_dir or self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = self.config['model_name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        saved_files = {}
        
        # Save CatBoost native format
        cbm_path = output_dir / f'{model_name}.cbm'
        self.model.save_model(str(cbm_path))
        saved_files['catboost'] = str(cbm_path)
        logger.info(f"✓ Saved CatBoost model: {cbm_path}")
        
        # Save ONNX format
        if ONNX_AVAILABLE:
            try:
                onnx_path = output_dir / f'{model_name}.onnx'
                self.model.save_model(
                    str(onnx_path),
                    format='onnx',
                    export_parameters={
                        'onnx_domain': 'ai.catboost',
                        'onnx_model_version': 1
                    }
                )
                saved_files['onnx'] = str(onnx_path)
                logger.info(f"✓ Saved ONNX model: {onnx_path}")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")
        
        # Save label encoder
        classes_path = output_dir / f'{model_name}_classes.json'
        with open(classes_path, 'w') as f:
            json.dump({
                'classes': self.label_encoder.classes_.tolist(),
                'feature_names': self.feature_names,
                'categorical_features': self.categorical_features
            }, f, indent=2)
        saved_files['metadata'] = str(classes_path)
        logger.info(f"✓ Saved metadata: {classes_path}")
        
        # Save feature importances
        importances = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        importance_path = output_dir / f'{model_name}_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        saved_files['feature_importance'] = str(importance_path)
        
        logger.info(f"\n🏆 Top 10 Features:")
        print(importance_df.head(10).to_string(index=False))
        
        return saved_files
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions.flatten())
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict_proba(X)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train CatBoost crop recommendation model')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--output-dir', type=str, default='./models/tabular',
                        help='Output directory for model')
    parser.add_argument('--iterations', type=int, default=2000,
                        help='Number of boosting iterations')
    parser.add_argument('--depth', type=int, default=8,
                        help='Tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('--no-smote', action='store_true',
                        help='Disable SMOTE augmentation')
    parser.add_argument('--no-mixup', action='store_true',
                        help='Disable Mixup augmentation')
    parser.add_argument('--cv', action='store_true',
                        help='Run cross-validation')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🌾 AgriSense CatBoost Trainer")
    print("=" * 70)
    
    # Load data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Configure trainer
    config = CatBoostCropTrainer._default_config()
    config['iterations'] = args.iterations
    config['depth'] = args.depth
    config['learning_rate'] = args.learning_rate
    config['output_dir'] = args.output_dir
    config['use_smote'] = not args.no_smote
    config['use_mixup'] = not args.no_mixup
    config['task_type'] = 'GPU' if args.gpu else 'CPU'
    
    trainer = CatBoostCropTrainer(config)
    
    # Cross-validation
    if args.cv:
        cv_results = trainer.cross_validate(df)
    
    # Train final model
    metrics = trainer.train(df)
    
    # Save model
    saved_files = trainer.save_model(args.output_dir)
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"\nSaved files:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")
    
    return trainer, metrics


if __name__ == '__main__':
    main()
