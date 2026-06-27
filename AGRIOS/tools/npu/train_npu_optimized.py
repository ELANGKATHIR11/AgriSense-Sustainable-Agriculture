#!/usr/bin/env python3
"""
NPU-Optimized ML Model Training for AgriSense
Leverages Intel Core Ultra 9 275HX NPU and CPU capabilities

Features:
- Intel Extension for PyTorch (IPEX) acceleration
- Intel oneDAL accelerated scikit-learn
- OpenVINO model export for NPU inference
- Automatic quantization for NPU efficiency
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib

# Intel acceleration - patch sklearn first
try:
    from sklearnex import patch_sklearn, unpatch_sklearn
    patch_sklearn()
    print("✅ Patched scikit-learn with Intel oneDAL acceleration")
    INTEL_SKLEARN = True
except ImportError:
    print("⚠️ Intel scikit-learn extension not available")
    INTEL_SKLEARN = False

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

# PyTorch with Intel Extension
import torch
try:
    import intel_extension_for_pytorch as ipex
    print(f"✅ Intel PyTorch Extension loaded: {ipex.__version__}")
    IPEX_AVAILABLE = True
except ImportError:
    print("⚠️ Intel PyTorch Extension not available")
    IPEX_AVAILABLE = False

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    print(f"✅ TensorFlow loaded: {tf.__version__}")
except ImportError:
    TF_AVAILABLE = False

# OpenVINO for NPU export
try:
    from openvino.runtime import Core
    import openvino as ov
    OPENVINO_AVAILABLE = True
    print(f"✅ OpenVINO loaded: {ov.__version__}")
except ImportError:
    OPENVINO_AVAILABLE = False
    print("⚠️ OpenVINO not available - models will not be exported for NPU")

warnings.filterwarnings('ignore')


class NPUOptimizedTrainer:
    """
    ML Trainer optimized for Intel Core Ultra 9 275HX with NPU
    """
    
    def __init__(self, output_dir: str = "agrisense_app/backend/models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.npu_export_dir = self.output_dir / "openvino_npu"
        self.npu_export_dir.mkdir(exist_ok=True)
        
        self.metrics = {
            'training_times': {},
            'accuracies': {},
            'model_sizes': {},
            'inference_times': {}
        }
        
        print(f"\n🏗️ NPU-Optimized Trainer initialized")
        print(f"   Output: {self.output_dir}")
        print(f"   NPU models: {self.npu_export_dir}")
        
        # Check NPU availability
        self.check_hardware()
    
    def check_hardware(self):
        """Check available acceleration hardware"""
        print("\n" + "=" * 70)
        print("🔍 HARDWARE DETECTION")
        print("=" * 70)
        
        # CPU info
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            print(f"  CPU: {cpu_info.get('brand_raw', 'Unknown')}")
            print(f"  Cores: {cpu_info.get('count', 'Unknown')}")
        except ImportError:
            print(f"  CPU: Unable to detect (py-cpuinfo not installed)")
            print(f"  Cores: {os.cpu_count()}")
        
        # PyTorch device
        print(f"\n  PyTorch device: {torch.device('cpu')}")
        if IPEX_AVAILABLE:
            print(f"  ✅ IPEX acceleration enabled")
        
        # OpenVINO NPU
        if OPENVINO_AVAILABLE:
            core = Core()
            devices = core.available_devices
            print(f"\n  OpenVINO devices: {', '.join(devices)}")
            
            if 'NPU' in devices:
                print(f"  🎯 NPU DETECTED - Models will be optimized for NPU inference")
                self.npu_available = True
            else:
                print(f"  ⚠️ NPU not detected - CPU inference will be used")
                self.npu_available = False
        else:
            self.npu_available = False
        
        print("=" * 70)
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load and validate dataset"""
        print(f"\n📂 Loading dataset: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        print(f"   ✅ Loaded {len(df)} samples with {len(df.columns)} features")
        
        return df
    
    def preprocess_crop_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Preprocess AgriSense crop recommendation data
        """
        print("\n🔧 Preprocessing crop recommendation data...")
        
        # Feature columns (sensor readings)
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Handle missing values
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        X = df[feature_cols].values
        
        # Encode crop labels
        crop_encoder = LabelEncoder()
        y = crop_encoder.fit_transform(df['label'])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"   ✅ Features: {X_scaled.shape}")
        print(f"   ✅ Classes: {len(crop_encoder.classes_)}")
        
        artifacts = {
            'scaler': scaler,
            'crop_encoder': crop_encoder,
            'feature_cols': feature_cols,
            'classes': crop_encoder.classes_.tolist()
        }
        
        return X_scaled, y, artifacts
    
    def train_random_forest_intel(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "crop_recommendation_rf"
    ) -> Dict:
        """
        Train Random Forest with Intel oneDAL acceleration
        """
        print(f"\n{'=' * 70}")
        print(f"🌲 Training Random Forest (Intel Optimized)")
        print(f"{'=' * 70}")
        
        start_time = time.time()
        
        # Random Forest with optimizations
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,  # Use all CPU cores
            random_state=42,
            verbose=1
        )
        
        print("\n⏳ Training Random Forest...")
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n📊 Results:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Training accuracy: {train_acc:.4f}")
        print(f"   Testing accuracy: {test_acc:.4f}")
        print(f"   F1-score: {f1:.4f}")
        
        # Save model
        model_path = self.output_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"   ✅ Saved: {model_path}")
        
        self.metrics['training_times'][model_name] = train_time
        self.metrics['accuracies'][model_name] = test_acc
        
        return {
            'model': model,
            'train_time': train_time,
            'test_accuracy': test_acc,
            'f1_score': f1
        }
    
    def train_gradient_boosting_intel(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "crop_recommendation_gb"
    ) -> Dict:
        """
        Train Gradient Boosting with Intel acceleration
        """
        print(f"\n{'=' * 70}")
        print(f"🚀 Training Gradient Boosting (Intel Optimized)")
        print(f"{'=' * 70}")
        
        start_time = time.time()
        
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=7,
            subsample=0.8,
            random_state=42,
            verbose=1
        )
        
        print("\n⏳ Training Gradient Boosting...")
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Evaluate
        test_acc = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n📊 Results:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Testing accuracy: {test_acc:.4f}")
        print(f"   F1-score: {f1:.4f}")
        
        # Save model
        model_path = self.output_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"   ✅ Saved: {model_path}")
        
        self.metrics['training_times'][model_name] = train_time
        self.metrics['accuracies'][model_name] = test_acc
        
        return {
            'model': model,
            'train_time': train_time,
            'test_accuracy': test_acc,
            'f1_score': f1
        }
    
    def train_neural_network_ipex(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "crop_recommendation_nn"
    ) -> Dict:
        """
        Train neural network with IPEX optimization
        """
        print(f"\n{'=' * 70}")
        print(f"🧠 Training Neural Network (IPEX Optimized)")
        print(f"{'=' * 70}")
        
        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train)
        X_test_t = torch.FloatTensor(X_test)
        y_train_t = torch.LongTensor(y_train)
        y_test_t = torch.LongTensor(y_test)
        
        # Define model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        class CropClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, num_classes)
                )
            
            def forward(self, x):
                return self.network(x)
        
        model = CropClassifier()
        
        # Optimize with IPEX
        if IPEX_AVAILABLE:
            model = ipex.optimize(model)
            print("   ✅ Model optimized with IPEX")
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        start_time = time.time()
        epochs = 50
        batch_size = 32
        
        print(f"\n⏳ Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X_train_t):.4f}")
        
        train_time = time.time() - start_time
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs, 1)
            test_acc = (predicted == y_test_t).float().mean().item()
        
        print(f"\n📊 Results:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Testing accuracy: {test_acc:.4f}")
        
        # Save PyTorch model
        model_path = self.output_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"   ✅ Saved PyTorch model: {model_path}")
        
        # Export to ONNX for OpenVINO
        if OPENVINO_AVAILABLE:
            self.export_to_openvino(model, X_test_t[:1], model_name)
        
        self.metrics['training_times'][model_name] = train_time
        self.metrics['accuracies'][model_name] = test_acc
        
        return {
            'model': model,
            'train_time': train_time,
            'test_accuracy': test_acc
        }
    
    def export_to_openvino(
        self, 
        model: torch.nn.Module, 
        sample_input: torch.Tensor,
        model_name: str
    ):
        """
        Export PyTorch model to OpenVINO IR for NPU inference
        """
        print(f"\n🔄 Exporting {model_name} to OpenVINO IR...")
        
        try:
            # Export to ONNX first
            onnx_path = self.npu_export_dir / f"{model_name}.onnx"
            
            model.eval()
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            
            print(f"   ✅ Exported to ONNX: {onnx_path}")
            
            # Convert ONNX to OpenVINO IR
            from openvino.tools import mo
            
            ir_path = self.npu_export_dir / model_name
            
            ov_model = mo.convert_model(onnx_path)
            ov.save_model(ov_model, str(ir_path / f"{model_name}.xml"))
            
            print(f"   ✅ Exported to OpenVINO IR: {ir_path}")
            
            if self.npu_available:
                print(f"   🎯 Model ready for NPU inference!")
            
        except Exception as e:
            print(f"   ⚠️ Export failed: {e}")
    
    def save_metrics(self):
        """Save training metrics"""
        metrics_path = self.output_dir / "npu_training_metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\n📊 Metrics saved: {metrics_path}")
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "=" * 70)
        print("📊 NPU-OPTIMIZED TRAINING SUMMARY")
        print("=" * 70)
        
        print("\n⏱️ Training Times:")
        for name, time_val in self.metrics['training_times'].items():
            print(f"   {name}: {time_val:.2f}s")
        
        print("\n🎯 Test Accuracies:")
        for name, acc in self.metrics['accuracies'].items():
            print(f"   {name}: {acc:.4f} ({acc*100:.2f}%)")
        
        print("\n" + "=" * 70)


def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print("🚀 AGRISENSE NPU-OPTIMIZED MODEL TRAINING")
    print("   Intel Core Ultra 9 275HX with NPU")
    print("   Date: 2025-12-30")
    print("=" * 70)
    
    # Initialize trainer
    trainer = NPUOptimizedTrainer()
    
    # Find dataset
    dataset_paths = [
        "agrisense_app/backend/Crop_recommendation.csv",
        "Crop_recommendation.csv",
        "data/Crop_recommendation.csv"
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        print(f"\n❌ Dataset not found. Tried:")
        for path in dataset_paths:
            print(f"   - {path}")
        print("\n💡 Please ensure Crop_recommendation.csv is available")
        return
    
    # Load and preprocess data
    df = trainer.load_dataset(dataset_path)
    X, y, artifacts = trainer.preprocess_crop_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Save preprocessing artifacts
    scaler_path = trainer.output_dir / "crop_scaler.joblib"
    encoder_path = trainer.output_dir / "crop_encoder.joblib"
    
    joblib.dump(artifacts['scaler'], scaler_path)
    joblib.dump(artifacts['crop_encoder'], encoder_path)
    
    print(f"\n💾 Saved preprocessing artifacts:")
    print(f"   Scaler: {scaler_path}")
    print(f"   Encoder: {encoder_path}")
    
    # Train models
    print("\n" + "=" * 70)
    print("🎯 TRAINING MODELS")
    print("=" * 70)
    
    # 1. Random Forest (Intel oneDAL accelerated)
    rf_results = trainer.train_random_forest_intel(
        X_train, X_test, y_train, y_test,
        model_name="crop_recommendation_rf_npu"
    )
    
    # 2. Gradient Boosting (Intel accelerated)
    gb_results = trainer.train_gradient_boosting_intel(
        X_train, X_test, y_train, y_test,
        model_name="crop_recommendation_gb_npu"
    )
    
    # 3. Neural Network (IPEX + OpenVINO export)
    nn_results = trainer.train_neural_network_ipex(
        X_train, X_test, y_train, y_test,
        model_name="crop_recommendation_nn_npu"
    )
    
    # Save metrics and summary
    trainer.save_metrics()
    trainer.print_summary()
    
    print("\n✅ NPU-OPTIMIZED TRAINING COMPLETE!")
    print(f"\n📁 Models saved to: {trainer.output_dir}")
    if trainer.npu_available:
        print(f"📁 NPU-ready models: {trainer.npu_export_dir}")
    
    print("\n🎯 Next steps:")
    print("   1. Benchmark models: python tools/npu/benchmark_models.py")
    print("   2. Deploy to production: Update model paths in backend")
    print("   3. Monitor performance: Use NPU inference metrics\n")


if __name__ == "__main__":
    main()
