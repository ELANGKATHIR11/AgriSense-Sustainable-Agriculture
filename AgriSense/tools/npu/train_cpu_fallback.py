"""
CPU-Optimized Model Training (Python 3.14 Fallback)
====================================================

This script trains AgriSense ML models using standard libraries
when NPU packages aren't compatible with Python 3.14.

Uses Intel-optimized scikit-learn (oneDAL) if available.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Try to use Intel-accelerated sklearn if available
try:
    from sklearnex import patch_sklearn  # type: ignore[import-not-found]
    patch_sklearn()
    print("✅ Using Intel-accelerated scikit-learn (oneDAL)")
    INTEL_OPTIMIZED = True
except ImportError:
    print("⚠️ Intel scikit-learn acceleration not available, using standard sklearn")
    INTEL_OPTIMIZED = False


class AgriSenseTrainer:
    """Optimized ML model trainer for AgriSense crop recommendation"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.models_dir = Path("agrisense_app/backend/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.scaler: StandardScaler | None = None
        self.label_encoder: LabelEncoder | None = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess crop recommendation dataset"""
        print("\n" + "=" * 70)
        print("📥 LOADING DATA")
        print("=" * 70)
        
        df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(df)} samples")
        print(f"Features: {', '.join(df.columns[:-1])}")
        print(f"Classes: {df['label'].nunique()}")
        
        # Separate features and labels
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, X.columns
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest with Intel optimizations"""
        print("\n" + "=" * 70)
        print("🌲 TRAINING RANDOM FOREST")
        print("=" * 70)
        
        start_time = time.time()
        
        # Configure for Intel CPU optimization
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,  # Use all CPU cores
            random_state=42,
            verbose=1
        )
        
        print(f"Configuration: {model.get_params()}")
        print(f"Using {'Intel oneDAL' if INTEL_OPTIMIZED else 'standard'} backend")
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        
        print(f"\n✅ Training completed in {training_time:.2f}s")
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # Save model
        model_path = self.models_dir / "random_forest_optimized.pkl"
        joblib.dump(model, model_path)
        print(f"💾 Model saved to: {model_path}")
        
        return model, {
            "training_time": training_time,
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "predictions": y_pred
        }
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting with Intel optimizations"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING GRADIENT BOOSTING")
        print("=" * 70)
        
        start_time = time.time()
        
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
            verbose=1
        )
        
        print(f"Configuration: {model.get_params()}")
        print(f"Using {'Intel oneDAL' if INTEL_OPTIMIZED else 'standard'} backend")
        
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        
        print(f"\n✅ Training completed in {training_time:.2f}s")
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # Save model
        model_path = self.models_dir / "gradient_boosting_optimized.pkl"
        joblib.dump(model, model_path)
        print(f"💾 Model saved to: {model_path}")
        
        return model, {
            "training_time": training_time,
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "predictions": y_pred
        }
    
    def save_artifacts(self):
        """Save preprocessing artifacts"""
        print("\n" + "=" * 70)
        print("💾 SAVING PREPROCESSING ARTIFACTS")
        print("=" * 70)
        
        if self.scaler is None or self.label_encoder is None:
            raise ValueError("Models must be trained before saving artifacts. Call load_and_preprocess_data() first.")
        
        # Save scaler
        scaler_path = self.models_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved: {scaler_path}")
        
        # Save label encoder
        encoder_path = self.models_dir / "label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        print(f"✅ Label encoder saved: {encoder_path}")
        
        # Save label mapping
        label_mapping = {
            int(i): label for i, label in enumerate(self.label_encoder.classes_)
        }
        mapping_path = self.models_dir / "label_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        print(f"✅ Label mapping saved: {mapping_path}")
    
    def generate_report(self, rf_results, gb_results):
        """Generate training report"""
        print("\n" + "=" * 70)
        print("📊 TRAINING SUMMARY")
        print("=" * 70)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "intel_optimized": INTEL_OPTIMIZED,
            "random_forest": {
                "training_time_seconds": rf_results["training_time"],
                "train_accuracy": float(rf_results["train_accuracy"]),
                "test_accuracy": float(rf_results["test_accuracy"])
            },
            "gradient_boosting": {
                "training_time_seconds": gb_results["training_time"],
                "train_accuracy": float(gb_results["train_accuracy"]),
                "test_accuracy": float(gb_results["test_accuracy"])
            }
        }
        
        # Save report
        report_path = self.models_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'Backend:':<25} {'Intel oneDAL' if INTEL_OPTIMIZED else 'Standard scikit-learn'}")
        print(f"{'Python version:':<25} {sys.version.split()[0]}")
        print(f"\n{'Model':<25} {'Train Acc':<15} {'Test Acc':<15} {'Time (s)':<15}")
        print("-" * 70)
        print(f"{'Random Forest':<25} {rf_results['train_accuracy']:<15.4f} {rf_results['test_accuracy']:<15.4f} {rf_results['training_time']:<15.2f}")
        print(f"{'Gradient Boosting':<25} {gb_results['train_accuracy']:<15.4f} {gb_results['test_accuracy']:<15.4f} {gb_results['training_time']:<15.2f}")
        print("\n✅ Report saved to:", report_path)
        
        return report


def main():
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "AgriSense CPU-Optimized Training" + " " * 21 + "║")
    print("║" + " " * 17 + "Intel Core Ultra 9 275HX" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    # Find dataset (prefer synthesized Crop_recommendation if present)
    dataset_paths = [
        "agrisense_app/backend/data/Crop_recommendation.csv",
        "agrisense_app/backend/india_crop_dataset.csv",
        "india_crop_dataset.csv",
        "datasets/raw/sikkim_crop_dataset.csv"
    ]
    
    data_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            data_path = path
            print(f"✅ Found dataset: {path}")
            break
    
    if not data_path:
        print(f"\n❌ Dataset not found. Tried:")
        for path in dataset_paths:
            print(f"   - {path}")
        sys.exit(1)
    
    # Initialize trainer
    trainer = AgriSenseTrainer(data_path)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = trainer.load_and_preprocess_data()
    
    # Train models
    rf_model, rf_results = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    gb_model, gb_results = trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)
    
    # Save artifacts
    trainer.save_artifacts()
    
    # Generate report
    report = trainer.generate_report(rf_results, gb_results)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print("\nModels saved to: agrisense_app/backend/models/")
    print("  - random_forest_optimized.pkl")
    print("  - gradient_boosting_optimized.pkl")
    print("  - scaler.pkl")
    print("  - label_encoder.pkl")
    print("  - label_mapping.json")
    print("  - training_report.json")
    print("\n🚀 Models ready for deployment!")


if __name__ == "__main__":
    main()
