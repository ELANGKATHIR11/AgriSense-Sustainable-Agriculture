"""
AGRISENSE Irrigation Optimization Training — PatchTST Sequence Model in PyTorch
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from ml.patchtst_models import PatchTST

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "AgriSense-Dataset")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def build_sequences_by_zone(df, feature_cols, target_col, seq_len=5):
    """
    Groups data by agro_climatic_zone (or fallback dummy) and extracts sliding sequences of length seq_len.
    """
    X_seq = []
    y_seq = []
    
    # If zone doesn't exist, group by district or just group into sets of 500 rows to simulate time series
    group_col = 'agro_climatic_zone' if 'agro_climatic_zone' in df.columns else None
    
    if group_col:
        grouped = df.groupby(group_col)
    else:
        # Dummy group every 500 elements
        df['dummy_group'] = np.arange(len(df)) // 500
        grouped = df.groupby('dummy_group')
        
    for name, group in grouped:
        if len(group) >= seq_len:
            feat_vals = group[feature_cols].values
            target_vals = group[target_col].values
            for i in range(len(group) - seq_len + 1):
                seq_f = feat_vals[i : i + seq_len].T # [num_features, seq_len]
                X_seq.append(seq_f)
                y_seq.append(target_vals[i + seq_len - 1])
        else:
            feat_vals = group[feature_cols].values
            target_vals = group[target_col].values
            padding_len = seq_len - len(group)
            pad_feat = np.repeat(feat_vals[0:1], padding_len, axis=0)
            padded_feat = np.vstack([pad_feat, feat_vals])
            X_seq.append(padded_feat.T)
            y_seq.append(target_vals[-1])
            
    return np.array(X_seq), np.array(y_seq)

def train_irrigation_model():
    print("  Loading cleaned irrigation datasets...")
    train_path = os.path.join(CLEANED_DIR, "irrigation_train.csv")
    val_path = os.path.join(CLEANED_DIR, "irrigation_val.csv")
    
    if not os.path.exists(train_path):
        print("  Cleaned irrigation datasets not found, loading original indian_agriculture_ml_dataset.csv")
        raw_path = os.path.join(DATA_DIR, "indian_agriculture_ml_dataset.csv")
        df = pd.read_csv(raw_path)
        df = df.drop_duplicates()
        df['water_required_liters'] = df['water_required_m3'] * 1000.0 if 'water_required_m3' in df.columns else 0.0
        df = df[df['water_required_liters'] >= 0]
        df = df[df['water_required_liters'] < 15000]
        df['soil_moisture_pct'] = df['soil_moisture_pct'].clip(5.0, 95.0)
        df['humidity_pct'] = df['humidity_pct'].clip(10.0, 100.0)
        df['temperature_avg_c'] = df.get('temperature_avg_c', df.get('avg_temp_c', 25.0)).clip(5.0, 50.0)
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

    # Feature engineering
    for df_split in [train_df, val_df]:
        df_split['moisture_deficit'] = np.maximum(0, 45 - df_split['soil_moisture_pct'])
        df_split['heat_stress'] = (df_split['temperature_avg_c'] > 32).astype(float)
        df_split['drought_stress'] = (df_split['soil_moisture_pct'] < 25).astype(float)
        df_split['temp_humid_idx'] = df_split['temperature_avg_c'] * (1 - df_split['humidity_pct'] / 100)
        df_split['et_demand'] = df_split.get('et0_mm', 4.5) * df_split.get('kc_value', 1.0)
        
        # NPK
        df_split['nitrogen'] = df_split.get('soil_n', 40.0)
        df_split['phosphorus'] = df_split.get('soil_p', 40.0)
        df_split['potassium'] = df_split.get('soil_k', 40.0)

    feature_cols = ['soil_moisture_pct', 'temperature_avg_c', 'humidity_pct',
                    'moisture_deficit', 'heat_stress', 'drought_stress',
                    'temp_humid_idx', 'et_demand', 'nitrogen', 'phosphorus', 'potassium']

    # Scale numeric columns
    scaler = StandardScaler()
    train_feats_scaled = scaler.fit_transform(train_df[feature_cols])
    val_feats_scaled = scaler.transform(val_df[feature_cols])
    
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    train_df_scaled[feature_cols] = train_feats_scaled
    val_df_scaled[feature_cols] = val_feats_scaled

    # Build sequence data
    seq_len = 5
    X_train, y_train = build_sequences_by_zone(train_df_scaled, feature_cols, 'water_required_liters', seq_len=seq_len)
    X_val, y_val = build_sequences_by_zone(val_df_scaled, feature_cols, 'water_required_liters', seq_len=seq_len)
    
    # Log-transform target values to avoid gradient explosion during MSE loss calculation
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    print(f"  Irrigation sequences built. Train shape: {X_train.shape} | Val shape: {X_val.shape}")

    # Datasets and Loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_log, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val_log, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Instantiate PatchTST
    c_in = len(feature_cols)
    model = PatchTST(
        c_in=c_in,
        context_window=seq_len,
        patch_len=2,
        stride=1,
        d_model=32,
        n_heads=2,
        d_ff=64,
        num_layers=2,
        target_dim=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Training PatchTST model on device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs = 10
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_log = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred.squeeze(), batch_y)
                val_loss += loss.item()
                val_preds_log.extend(pred.squeeze().cpu().numpy())
                
        # Exponentiate predictions and targets back to real-world liters for evaluation
        val_preds = np.expm1(val_preds_log)
        val_preds = np.clip(val_preds, 0, None) # Clamp to prevent negative liters
        
        r2 = r2_score(y_val, val_preds)
        mae = mean_absolute_error(y_val, val_preds)
        
        print(f"  Epoch {epoch}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss (Log): {val_loss/len(val_loader):.4f} | R2 (Liters): {r2:.4f} | MAE (Liters): {mae:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "irrigation_prediction_patchtst.pth"))
            
    # Save metadata
    joblib.dump(scaler, os.path.join(MODELS_DIR, "irrigation_scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "irrigation_feature_cols.joblib"))
    print(f"  Saved modernized irrigation PatchTST model weights and metadata successfully.")
    
    return r2, mae

if __name__ == "__main__":
    train_irrigation_model()
