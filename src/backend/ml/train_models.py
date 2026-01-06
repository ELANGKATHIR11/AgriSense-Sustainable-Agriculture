"""
ML Model Training Pipeline
Trains all 5 models for AgriSense recommendation system
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent / "models"
ENCODERS_DIR = Path(__file__).parent.parent / "data" / "encoders"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ModelTrainer:
    """Train and evaluate ML models for AgriSense"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        
    def load_dataset(self, dataset_name):
        """Load a preprocessed dataset"""
        pkl_file = DATA_DIR / dataset_name / f"{dataset_name}_complete.pkl"
        
        if not pkl_file.exists():
            raise FileNotFoundError(f"Dataset not found: {pkl_file}")
        
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def train_crop_recommendation_model(self):
        """Train crop recommendation (96-class classification)"""
        print("\n" + "="*60)
        print("🌾 Training Crop Recommendation Model (96 classes)")
        print("="*60)
        
        dataset = self.load_dataset("crop_recommendation")
        X_train, X_test = dataset['X_train'], dataset['X_test']
        y_train, y_test = dataset['y_train'], dataset['y_test']
        
        # Train Random Forest (handles multi-class well)
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"✅ Training complete")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score (weighted): {f1:.4f}")
        print(f"   Classes: {len(np.unique(y_train))}")
        
        self.models['crop_recommendation'] = model
        self.metrics['crop_recommendation'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classes': int(len(np.unique(y_train)))
        }
        
        return model
    
    def train_crop_type_classification_model(self):
        """Train crop type classification (10 classes)"""
        print("\n" + "="*60)
        print("🌱 Training Crop Type Classification Model (10 classes)")
        print("="*60)
        
        dataset = self.load_dataset("crop_type_classification")
        X_train, X_test = dataset['X_train'], dataset['X_test']
        y_train, y_test = dataset['y_train'], dataset['y_test']
        
        # Train XGBoost-like with Gradient Boosting
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"✅ Training complete")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score (weighted): {f1:.4f}")
        print(f"   Classes: {len(np.unique(y_train))}")
        
        self.models['crop_type_classification'] = model
        self.metrics['crop_type_classification'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classes': int(len(np.unique(y_train)))
        }
        
        return model
    
    def train_growth_duration_model(self):
        """Train growth duration prediction (regression)"""
        print("\n" + "="*60)
        print("📅 Training Growth Duration Prediction Model")
        print("="*60)
        
        dataset = self.load_dataset("growth_duration")
        X_train, X_test = dataset['X_train'], dataset['X_test']
        y_train, y_test = dataset['y_train'], dataset['y_test']
        
        # Train Random Forest Regressor
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Training complete")
        print(f"   RMSE: {rmse:.4f} days")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Target range: {y_test.min():.0f}-{y_test.max():.0f} days")
        
        self.models['growth_duration'] = model
        self.metrics['growth_duration'] = {
            'rmse': float(rmse),
            'r2_score': float(r2),
            'target_range': [float(y_test.min()), float(y_test.max())]
        }
        
        return model
    
    def train_water_requirement_model(self):
        """Train water requirement prediction (regression)"""
        print("\n" + "="*60)
        print("💧 Training Water Requirement Prediction Model")
        print("="*60)
        
        dataset = self.load_dataset("water_requirement")
        X_train, X_test = dataset['X_train'], dataset['X_test']
        y_train, y_test = dataset['y_train'], dataset['y_test']
        
        # Train Gradient Boosting Regressor
        model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Training complete")
        print(f"   RMSE: {rmse:.4f} mm/day")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Target range: {y_test.min():.2f}-{y_test.max():.2f} mm/day")
        
        self.models['water_requirement'] = model
        self.metrics['water_requirement'] = {
            'rmse': float(rmse),
            'r2_score': float(r2),
            'target_range': [float(y_test.min()), float(y_test.max())]
        }
        
        return model
    
    def train_season_classification_model(self):
        """Train season classification (5 classes)"""
        print("\n" + "="*60)
        print("🌾 Training Season Classification Model (5 classes)")
        print("="*60)
        
        dataset = self.load_dataset("season_classification")
        X_train, X_test = dataset['X_train'], dataset['X_test']
        y_train, y_test = dataset['y_train'], dataset['y_test']
        
        # Train Support Vector Machine (good for smaller datasets)
        model = SVC(
            kernel='rbf',
            C=100,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"✅ Training complete")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score (weighted): {f1:.4f}")
        print(f"   Classes: {len(np.unique(y_train))}")
        print(f"   Classes: {sorted(np.unique(y_train).tolist())}")
        
        self.models['season_classification'] = model
        self.metrics['season_classification'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classes': int(len(np.unique(y_train))),
            'class_names': sorted(np.unique(y_train).tolist())
        }
        
        return model
    
    def train_intent_classifier(self):
        """Train intent classifier for RAG pipeline (SVM)"""
        print("\n" + "="*60)
        print("🎯 Training Intent Classifier for RAG Pipeline")
        print("="*60)
        
        # Create synthetic intent training data
        intents = {
            'weather': ['temperature', 'rainfall', 'weather', 'climate', 'season', 'monsoon', 'rain'],
            'disease': ['disease', 'pest', 'blight', 'fungal', 'bacterial', 'viral', 'infection'],
            'soil': ['soil', 'pH', 'nutrients', 'NPK', 'fertilizer', 'amendments', 'tilth'],
            'crop_recommendation': ['recommend', 'suitable', 'best', 'grow', 'cultivate', 'select', 'choice'],
            'pricing': ['price', 'market', 'cost', 'sell', 'buy', 'profit', 'rate']
        }
        
        # Create feature vectors using keyword matching
        X_intent = []
        y_intent = []
        
        for intent, keywords in intents.items():
            for keyword in keywords:
                X_intent.append([len(keyword), ord(keyword[0]), keyword.count('e')])
                y_intent.append(intent)
        
        X_intent = np.array(X_intent)
        y_intent = np.array(y_intent)
        
        # Scale features
        scaler = StandardScaler()
        X_intent_scaled = scaler.fit_transform(X_intent)
        
        # Train SVM
        model = SVC(kernel='linear', probability=True, random_state=42)
        model.fit(X_intent_scaled, y_intent)
        
        # Evaluate
        y_pred = model.predict(X_intent_scaled)
        accuracy = accuracy_score(y_intent, y_pred)
        
        print(f"✅ Training complete")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Intents: {list(intents.keys())}")
        
        self.models['intent_classifier'] = model
        self.scalers['intent_classifier'] = scaler
        self.metrics['intent_classifier'] = {
            'accuracy': float(accuracy),
            'intents': list(intents.keys())
        }
        
        return model
    
    def save_models(self):
        """Save all trained models"""
        print("\n" + "="*60)
        print("💾 Saving Trained Models")
        print("="*60)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = MODELS_DIR / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ {model_name}: {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = MODELS_DIR / f"{scaler_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"✅ {scaler_name} scaler: {scaler_path}")
        
        # Save metrics
        metrics_path = MODELS_DIR / "model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Model metrics: {metrics_path}")
        
        # Save model manifest
        manifest = {
            'models': list(self.models.keys()),
            'training_date': pd.Timestamp.now().isoformat(),
            'metrics': self.metrics
        }
        manifest_path = MODELS_DIR / "model_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"✅ Model manifest: {manifest_path}")
        
        return metrics_path
    
    def train_all_models(self):
        """Train all models"""
        print("\n🚀 STARTING ML MODEL TRAINING PIPELINE")
        print("="*60)
        
        try:
            # Train classification models
            self.train_crop_recommendation_model()
            self.train_crop_type_classification_model()
            self.train_season_classification_model()
            
            # Train regression models
            self.train_growth_duration_model()
            self.train_water_requirement_model()
            
            # Train intent classifier
            self.train_intent_classifier()
            
            # Save all models
            self.save_models()
            
            print("\n" + "="*60)
            print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
            print("="*60)
            print("\nSummary:")
            for model_name, metric in self.metrics.items():
                print(f"\n{model_name}:")
                for key, value in metric.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    trainer = ModelTrainer()
    success = trainer.train_all_models()
    exit(0 if success else 1)
