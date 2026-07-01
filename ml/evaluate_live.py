# -*- coding: utf-8 -*-
"""
AGRISENSE Real-Time Live Model Evaluation Script
Performs a full evaluation of all trained models on their validation sets and outputs accuracy and metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_absolute_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
DATA_DIR = os.path.join(BASE_DIR, "AgriSense-Dataset")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")

def evaluate_crop():
    print("Evaluating Crop Recommendation model...")
    model_path = os.path.join(MODELS_DIR, "crop_recommendation_catboost.joblib")
    val_path = os.path.join(CLEANED_DIR, "crop_rec_val.csv")
    encoder_path = os.path.join(MODELS_DIR, "crop_label_encoder.joblib")
    
    if not (os.path.exists(model_path) and os.path.exists(val_path)):
        return {"status": "missing_files"}
        
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    val_df = pd.read_csv(val_path)
    
    feature_cols = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall',
                    'npk_ratio', 'nk_ratio', 'ph_bin', 'temp_bin', 'humid_class', 'rain_class']
                    
    val_df['npk_ratio']   = (val_df['n'] + val_df['k']) / (val_df['p'] + 1)
    val_df['nk_ratio']    = val_df['n'] / (val_df['k'] + 1)
    val_df['ph_bin']      = pd.cut(val_df['ph'], bins=[0,5,5.5,6,6.5,7,7.5,8,14], labels=[0,1,2,3,4,5,6,7]).astype(int)
    val_df['temp_bin']    = pd.cut(val_df['temperature'], bins=[0,15,20,25,30,35,50], labels=[0,1,2,3,4,5]).astype(int)
    val_df['humid_class'] = (val_df['humidity'] > 70).astype(int)
    val_df['rain_class']  = pd.cut(val_df['rainfall'], bins=[0,50,100,200,300,3000], labels=[0,1,2,3,4]).astype(int)
    
    X_val = val_df[feature_cols].astype(float)
    y_val = le.transform(val_df['label'])
    
    preds = model.predict(X_val)
    if preds.ndim > 1:
        preds = preds.squeeze()
        
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='weighted')
    prec = precision_score(y_val, preds, average='weighted')
    rec = recall_score(y_val, preds, average='weighted')
    
    return {
        "status": "success",
        "accuracy": float(acc),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "samples_evaluated": len(val_df)
    }

def evaluate_fertilizer():
    print("Evaluating Fertilizer Recommendation model...")
    model_path = os.path.join(MODELS_DIR, "fertilizer_recommendation_catboost.joblib")
    val_path = os.path.join(CLEANED_DIR, "fertilizer_val.csv")
    encoder_path = os.path.join(MODELS_DIR, "fertilizer_label_encoder.joblib")
    
    if not (os.path.exists(model_path) and os.path.exists(val_path)):
        return {"status": "missing_files"}
        
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    val_df = pd.read_csv(val_path)
    
    # Feature engineering matching training logic
    feature_cols = ['temperature', 'humidity', 'moisture', 'soil_type', 'crop_type', 'nitrogen', 'potassium', 'phosphorus']
    X_val = val_df[feature_cols]
    y_val = le.transform(val_df['fertilizer_name'])
    
    preds = model.predict(X_val)
    if preds.ndim > 1:
        preds = preds.squeeze()
        
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='weighted')
    prec = precision_score(y_val, preds, average='weighted')
    rec = recall_score(y_val, preds, average='weighted')
    
    return {
        "status": "success",
        "accuracy": float(acc),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "samples_evaluated": len(val_df)
    }

def evaluate_disease_risk():
    print("Evaluating Disease Risk model...")
    model_path = os.path.join(MODELS_DIR, "disease_risk_model.joblib")
    if not os.path.exists(model_path):
        return {"status": "missing_files"}
        
    bundle = joblib.load(model_path)
    
    from ml.disease_risk_training import load_disease_data
    from sklearn.model_selection import train_test_split
    df = load_disease_data()
    feature_cols = ['temperature_c', 'humidity_pct', 'leaf_wetness_hours', 'soil_moisture_pct', 'rainfall_mm']
    X = df[feature_cols].astype(float)
    y = df['risk_label'].astype(int)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    ensemble_model = bundle['model']
    scaler = bundle['scaler']
    
    X_val_scaled = scaler.transform(X_val)
    preds = ensemble_model.predict(X_val_scaled)
    
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='weighted')
    prec = precision_score(y_val, preds, average='weighted')
    rec = recall_score(y_val, preds, average='weighted')
    
    return {
        "status": "success",
        "accuracy": float(acc),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "samples_evaluated": len(y_val)
    }

def evaluate_irrigation():
    print("Evaluating Irrigation Optimization PatchTST model...")
    val_path = os.path.join(CLEANED_DIR, "irrigation_val.csv")
    model_path = os.path.join(MODELS_DIR, "irrigation_prediction_patchtst.pth")
    scaler_path = os.path.join(MODELS_DIR, "irrigation_scaler.joblib")
    
    if not (os.path.exists(model_path) and os.path.exists(val_path)):
        return {"status": "missing_files"}
        
    from ml.patchtst_models import PatchTST
    from ml.irrigation_training import build_sequences_by_zone
    
    val_df = pd.read_csv(val_path)
    scaler = joblib.load(scaler_path)
    
    val_df['moisture_deficit'] = np.maximum(0, 45 - val_df['soil_moisture_pct'])
    val_df['heat_stress'] = (val_df['temperature_avg_c'] > 32).astype(float)
    val_df['drought_stress'] = (val_df['soil_moisture_pct'] < 25).astype(float)
    val_df['temp_humid_idx'] = val_df['temperature_avg_c'] * (1 - val_df['humidity_pct'] / 100)
    val_df['et_demand'] = val_df.get('et0_mm', 4.5) * val_df.get('kc_value', 1.0)
    val_df['nitrogen'] = val_df.get('soil_n', 40.0)
    val_df['phosphorus'] = val_df.get('soil_p', 40.0)
    val_df['potassium'] = val_df.get('soil_k', 40.0)
    
    feature_cols = ['soil_moisture_pct', 'temperature_avg_c', 'humidity_pct',
                    'moisture_deficit', 'heat_stress', 'drought_stress',
                    'temp_humid_idx', 'et_demand', 'nitrogen', 'phosphorus', 'potassium']
                    
    val_df_scaled = val_df.copy()
    val_df_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    
    # Build sequences
    X_val, y_val = build_sequences_by_zone(val_df_scaled, feature_cols, 'water_required_liters', seq_len=5)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchTST(
        c_in=len(feature_cols),
        context_window=5,
        patch_len=2,
        stride=1,
        d_model=32,
        n_heads=2,
        d_ff=64,
        num_layers=2,
        target_dim=1
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
        preds = model(inputs).cpu().numpy().squeeze()
        
    r2 = r2_score(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    
    return {
        "status": "success",
        "r2_score": float(r2),
        "mae": float(mae),
        "samples_evaluated": len(X_val)
    }

def evaluate_yield():
    print("Evaluating Yield Prediction PatchTST model...")
    val_path = os.path.join(CLEANED_DIR, "yield_val.csv")
    model_path = os.path.join(MODELS_DIR, "yield_prediction_patchtst.pth")
    encoders_path = os.path.join(MODELS_DIR, "yield_encoders.joblib")
    scaler_path = os.path.join(MODELS_DIR, "yield_scaler.joblib")
    
    if not (os.path.exists(model_path) and os.path.exists(val_path)):
        return {"status": "missing_files"}
        
    from ml.patchtst_models import PatchTST
    from ml.yield_training import build_sequences
    
    val_df = pd.read_csv(val_path)
    encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)
    
    le_crop = encoders['Crop']
    le_season = encoders['Season']
    le_state = encoders['State']
    
    val_df['crop_enc'] = le_crop.transform(val_df['crop'].astype(str))
    val_df['season_enc'] = le_season.transform(val_df['season'].astype(str))
    val_df['state_enc'] = le_state.transform(val_df['state'].astype(str))
    
    val_df['area'] = pd.to_numeric(val_df['area'], errors='coerce').fillna(5.0).clip(0.01, 1e6)
    val_df['annual_rainfall'] = pd.to_numeric(val_df['annual_rainfall'], errors='coerce').fillna(150.0).clip(0, 5000)
    val_df['fertilizer'] = pd.to_numeric(val_df['fertilizer'], errors='coerce').fillna(100.0).clip(0, 50000)
    val_df['pesticide'] = pd.to_numeric(val_df['pesticide'], errors='coerce').fillna(10.0).clip(0, 1000)
    
    val_df['nitrogen'] = 40.0
    val_df['phosphorus'] = 40.0
    val_df['potassium'] = 40.0
    val_df['npk_total'] = 120.0
    val_df['rain_per_ha'] = val_df['annual_rainfall'] / (val_df['area'] + 1)
    val_df['fert_per_ha'] = val_df['fertilizer'] / (val_df['area'] + 1)
    
    feature_cols = ['crop_enc', 'season_enc', 'state_enc', 'area', 'annual_rainfall', 'fertilizer', 'pesticide',
                    'nitrogen', 'phosphorus', 'potassium', 'npk_total', 'rain_per_ha', 'fert_per_ha']
                    
    val_df_scaled = val_df.copy()
    val_df_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    
    X_val, y_val = build_sequences(val_df_scaled, feature_cols, 'yield', seq_len=5)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchTST(
        c_in=len(feature_cols),
        context_window=5,
        patch_len=2,
        stride=1,
        d_model=32,
        n_heads=2,
        d_ff=64,
        num_layers=2,
        target_dim=1
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
        preds = model(inputs).cpu().numpy().squeeze()
        
    r2 = r2_score(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    
    return {
        "status": "success",
        "r2_score": float(r2),
        "mae": float(mae),
        "samples_evaluated": len(X_val)
    }

def main():
    print("=" * 60)
    print("             AGRISENSE LIVE MODEL EVALUATION")
    print("=" * 60)
    
    results = {}
    
    results["crop_recommendation"] = evaluate_crop()
    results["fertilizer_recommendation"] = evaluate_fertilizer()
    results["disease_risk"] = evaluate_disease_risk()
    results["irrigation"] = evaluate_irrigation()
    results["yield_prediction"] = evaluate_yield()
    
    print("\n" + "=" * 60)
    print("                      RESULTS SUMMARY")
    print("=" * 60)
    
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        if metrics["status"] == "success":
            for k, v in metrics.items():
                if k != "status":
                    print(f"  {k}: {v}")
        else:
            print(f"  Status: {metrics['status']}")
            
    # Save the metrics to a centralized MLOps evaluation metrics file
    out_path = os.path.join(MODELS_DIR, "live_evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation metrics to {out_path}")

if __name__ == "__main__":
    main()
