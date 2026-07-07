# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

"""
AGRISENSE Fertilizer Recommendation Training — CatBoost Classifier
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "AgriSense-Dataset", "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_fertilizer_model():
    print("  Loading cleaned fertilizer datasets...")
    train_path = os.path.join(DATA_DIR, "fertilizer_train.csv")
    val_path = os.path.join(DATA_DIR, "fertilizer_val.csv")
    
    if not os.path.exists(train_path):
        print("  Cleaned data not found, loading raw fertilizer data")
        raw_path = os.path.join(BASE_DIR, "AgriSense-Dataset", "fertilizer_dataset.csv")
        df = pd.read_csv(raw_path)
        rename_map = {
            'Temparature': 'temperature', 'Humidity ': 'humidity', 'Moisture': 'moisture',
            'Soil Type': 'soil_type', 'Crop Type': 'crop_type',
            'Nitrogen': 'nitrogen', 'Potassium': 'potassium', 'Phosphorous': 'phosphorus',
            'Fertilizer Name': 'fertilizer_name'
        }
        df = df.rename(columns=rename_map)
        df.columns = df.columns.str.strip().str.lower()
        df = df.drop_duplicates().dropna()
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
    feature_cols = ['temperature', 'humidity', 'moisture', 'soil_type', 'crop_type', 'nitrogen', 'potassium', 'phosphorus']
    cat_features = ['soil_type', 'crop_type']
    
    X_train = train_df[feature_cols]
    y_train = train_df['fertilizer_name']
    X_val = val_df[feature_cols]
    y_val = val_df['fertilizer_name']

    le = LabelEncoder()
    # Fit on combined target column to handle unseen validation categories
    le.fit(pd.concat([y_train, y_val]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    print("  Training CatBoost Fertilizer Classifier...")
    task_type = "GPU" if torch.cuda.is_available() else "CPU"
    
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=5,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        cat_features=cat_features,
        task_type=task_type,
        random_seed=42,
        verbose=100
    )
    
    model.fit(
        X_train, y_train_enc,
        eval_set=(X_val, y_val_enc),
        early_stopping_rounds=20,
        use_best_model=True
    )

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val_enc, y_pred)
    f1 = f1_score(y_val_enc, y_pred, average='weighted')

    print(f"  Validation Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")
    
    # Save model and encoder
    joblib.dump(model, os.path.join(MODELS_DIR, "fertilizer_recommendation_catboost.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "fertilizer_label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "fertilizer_feature_cols.joblib"))
    print(f"  Saved modernized fertilizer model to {MODELS_DIR}/fertilizer_recommendation_catboost.joblib")

    return acc, f1

if __name__ == "__main__":
    train_fertilizer_model()
