"""
AGRISENSE Crop Recommendation Training — CatBoost Classifier
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from catboost import CatBoostClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "AgriSense-Dataset", "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_crop_model():
    print("  Loading cleaned crop datasets...")
    train_path = os.path.join(DATA_DIR, "crop_rec_train.csv")
    val_path = os.path.join(DATA_DIR, "crop_rec_val.csv")
    
    if not os.path.exists(train_path):
        print("  Cleaned data not found, falling back to original path")
        raw_path = os.path.join(BASE_DIR, "AgriSense-Dataset", "Crop_recommendation.csv")
        df = pd.read_csv(raw_path)
        df.columns = df.columns.str.strip().str.lower()
        df = df.drop_duplicates().dropna()
        median_rf = df['rainfall'].median()
        train_df = df[df['rainfall'] <= median_rf * 1.25]
        val_df = df[df['rainfall'] > median_rf * 1.25]
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
    feature_cols = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall',
                    'npk_ratio', 'nk_ratio', 'ph_bin', 'temp_bin', 'humid_class', 'rain_class']
                    
    for df_split in [train_df, val_df]:
        df_split['npk_ratio']   = (df_split['n'] + df_split['k']) / (df_split['p'] + 1)
        df_split['nk_ratio']    = df_split['n'] / (df_split['k'] + 1)
        df_split['ph_bin']      = pd.cut(df_split['ph'], bins=[0,5,5.5,6,6.5,7,7.5,8,14], labels=[0,1,2,3,4,5,6,7]).astype(int)
        df_split['temp_bin']    = pd.cut(df_split['temperature'], bins=[0,15,20,25,30,35,50], labels=[0,1,2,3,4,5]).astype(int)
        df_split['humid_class'] = (df_split['humidity'] > 70).astype(int)
        df_split['rain_class']  = pd.cut(df_split['rainfall'], bins=[0,50,100,200,300,3000], labels=[0,1,2,3,4]).astype(int)

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df['label']
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df['label']

    le = LabelEncoder()
    # Fit on both train and validation labels to support all classes in non-random splits
    le.fit(pd.concat([y_train, y_val]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    
    n_classes = len(le.classes_)
    print(f"  Classes: {n_classes} crops | Train: {len(X_train)} | Val: {len(X_val)}")

    print("  Training CatBoost Classifier...")
    task_type = "GPU" if torch.cuda.is_available() else "CPU"
    
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.08,
        depth=6,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        task_type=task_type,
        verbose=100
    )
    
    model.fit(
        X_train, y_train_enc,
        eval_set=(X_val, y_val_enc),
        early_stopping_rounds=30,
        use_best_model=True
    )

    y_val_unique = np.unique(y_val_enc)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val_enc, y_pred)
    f1 = f1_score(y_val_enc, y_pred, average='weighted')

    print(f"  Validation Accuracy: {acc*100:.2f}%")
    print(f"  Weighted F1:   {f1:.4f}")
    
    joblib.dump(model, os.path.join(MODELS_DIR, "crop_recommendation_catboost.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "crop_label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "crop_feature_cols.joblib"))
    print(f"  Saved modernized crop model to {MODELS_DIR}/crop_recommendation_catboost.joblib")

    return acc, f1

import torch
if __name__ == "__main__":
    train_crop_model()
