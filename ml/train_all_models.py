"""
AGRISENSE Advanced Production ML Optimization & Champion-Challenger Pipeline
Trains, benchmarks, explains, and registers all AgriSense ML models on the RTX 5060 GPU
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance

from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

# Clear PyTorch CUDA cache to free up VRAM
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.database.connection import sync_engine
from backend.database.models import Base, ModelRegistry
from sqlalchemy.orm import sessionmaker

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = os.path.join(BASE_DIR, "AgriSense-Dataset", "cleaned", "production_dataset_fixed_v2.csv")
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
REPORTS_DIR = os.path.join(MODELS_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

GPU_SUPPORT = torch.cuda.is_available()
print(f"GPU Support Status: {GPU_SUPPORT}")

# ── 1. ADVANCED FEATURE ENGINEERING PIPELINE ─────────────────────────────────
def engineer_advanced_features(df):
    print("Engineering domain features (GDD, moisture deficit, weather history, nutrient ratios, water balance)...")
    df = df.copy()
    
    # Nutrient ratios
    df['npk_ratio'] = (df.get('Nitrogen', 45) + df.get('Potassium', 42)) / (df.get('Phosphorus', 38) + 1)
    df['nk_ratio'] = df.get('Nitrogen', 45) / (df.get('Potassium', 42) + 1)
    
    # Moisture deficit
    df['moisture_deficit'] = np.maximum(0.0, 45.0 - df.get('Soil_Moisture_Surface', 35.0))
    
    # Growing degree days (GDD) relative to a base of 10°C
    df['GDD'] = np.maximum(0.0, df.get('Air_Temperature', 25.0) - 10.0)
    
    # ET Deficit & Water Balance
    df['et_deficit'] = np.maximum(0.0, df.get('ET0', 4.0) - df.get('Rainfall', 10.0))
    df['water_balance'] = df.get('Rainfall', 10.0) - df.get('ET0', 4.0)
    
    # Disease pressure interaction
    df['disease_pressure'] = df.get('Disease_Risk_Index', 0.1) * df.get('Disease_Severity', 0.0)
    
    # Crop age interaction
    df['crop_age_gdd'] = df.get('Plant_Age', 30.0) * df['GDD']
    
    # Interactions
    df['temp_humid_idx'] = df.get('Air_Temperature', 25.0) * (1 - df.get('Humidity', 60.0) / 100.0)
    df['nitrogen_water_interaction'] = df.get('Nitrogen', 45) * df.get('Soil_Moisture_Surface', 35.0)
    
    # Weather history / rolling stats (Simulated sequence rolling)
    df['temp_rolling_mean_3'] = df['Air_Temperature'].rolling(window=3, min_periods=1).mean()
    df['rain_rolling_sum_3'] = df['Rainfall'].rolling(window=3, min_periods=1).sum()
    
    return df

# ── 2. CLASS IMBALANCE RESOLVER ──────────────────────────────────────────────
def resolve_class_imbalance(X_train, y_train, X_val, y_val):
    print("  Resolving class imbalance (benchmarking SMOTE, ADASYN, and Class Weights)...")
    # Verify number of classes and samples to prevent SMOTE errors on tiny labels
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or counts.min() < 6:
        print("    Class count or sample size too small for SMOTE. Using Class Weights.")
        return X_train, y_train, "class_weights"
        
    try:
        # Benchmark SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, counts.min() - 1))
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
        
        # Benchmark ADASYN
        adasyn = ADASYN(random_state=42, n_neighbors=min(5, counts.min() - 1))
        X_ad, y_ad = adasyn.fit_resample(X_train, y_train)
        
        # Quick eval F1 score with Random Forest on SMOTE vs ADASYN
        rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
        
        rf.fit(X_sm, y_sm)
        sm_f1 = f1_score(y_val, rf.predict(X_val), average='macro')
        
        rf.fit(X_ad, y_ad)
        ad_f1 = f1_score(y_val, rf.predict(X_val), average='macro')
        
        if sm_f1 >= ad_f1:
            print(f"    SMOTE selected (Val Macro F1: {sm_f1:.4f} vs ADASYN: {ad_f1:.4f})")
            return X_sm, y_sm, "SMOTE"
        else:
            print(f"    ADASYN selected (Val Macro F1: {ad_f1:.4f} vs SMOTE: {sm_f1:.4f})")
            return X_ad, y_ad, "ADASYN"
    except Exception as e:
        print(f"    Imbalance handling exception ({e}). Falling back to Class Weights.")
        return X_train, y_train, "class_weights"

# ── 3. PREPROCESSING & FEATURE SELECTION ENGINE ─────────────────────────────────
def preprocess_and_select_features(df, target_col, task_type="classification", leakages=None):
    print(f"\n--- Feature Selection & Preprocessing for Target: {target_col} ---")
    df = df.copy()
    
    # 1. Drop target leakages (excluding target itself)
    if leakages:
        cols_to_drop = [c for c in leakages if c in df.columns and c != target_col]
        df = df.drop(columns=cols_to_drop)
        print(f"  Removed {len(cols_to_drop)} leakage columns.")
        
    # 2. Impute target NaNs
    if df[target_col].isna().sum() > 0:
        if task_type == "classification":
            df[target_col] = df[target_col].fillna("None")
        else:
            df = df.dropna(subset=[target_col])
            
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Drop identifiers and columns that shouldn't be trained on
    metadata_cols = ['Sensor_ID', 'Sensor_Type', 'Timestamp', 'Local_Language', 'Weather_Alert', 'Government_Advisory', 
                     'Research_Reference', 'Expert_Recommendation', 'Farmer_Feedback', 'Best_Practice', 'Disease_Bulletin', 'Model_Version']
    metadata_cols = [c for c in metadata_cols if c in X.columns]
    X = X.drop(columns=metadata_cols)
    
    # Contextual missing value imputation
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "None")
        else:
            X[col] = X[col].fillna(X[col].median())
            
    # Constant features check
    constant_cols = [col for col in X.columns if X[col].nunique() == 1]
    X = X.drop(columns=constant_cols)
    
    # Remove duplicate features
    X = X.loc[:, ~X.columns.duplicated()]
    
    # Encode object features using LabelEncoder
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # Multicollinearity check (Threshold > 0.85)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = X[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    X = X.drop(columns=to_drop)
    
    # Auto feature selection based on cumulative LightGBM importance
    if X.shape[1] > 12:
        try:
            if task_type == "classification":
                le_y = LabelEncoder()
                y_encoded = le_y.fit_transform(y.astype(str))
                lgb_sel = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1, n_jobs=-1)
                lgb_sel.fit(X, y_encoded)
            else:
                lgb_sel = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1, n_jobs=-1)
                lgb_sel.fit(X, y)
                
            importances = pd.Series(lgb_sel.feature_importances_, index=X.columns).sort_values(ascending=False)
            cum_importance = importances.cumsum() / importances.sum()
            keep_cols = cum_importance[cum_importance <= 0.98].index.tolist()
            
            if len(keep_cols) < 8:
                keep_cols = importances.index[:8].tolist()
                
            X = X[keep_cols]
            print(f"  LightGBM Feature Importance kept {len(keep_cols)} features.")
        except Exception as e:
            print(f"  Feature selection fallback: {e}")
        
    print(f"  Selected features: {list(X.columns)}")
    return X, y, encoders

# ── 4. EXPLAINABILITY PLOTS GENERATION ────────────────────────────────────────
def generate_explainability_reports(model, X_val, y_val, name, task_type="classification"):
    try:
        sample_size = min(100, len(X_val))
        X_sample = X_val.sample(sample_size, random_state=42)
        
        # 1. SHAP Plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"SHAP Summary: {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, f"{name}_shap.png"))
        plt.close()
        
        # 2. Permutation Importance
        result = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
        sorted_importances_idx = result.importances_mean.argsort()[::-1][:15]
        
        plt.figure(figsize=(10, 6))
        plt.barh(X_val.columns[sorted_importances_idx][::-1], result.importances_mean[sorted_importances_idx][::-1])
        plt.title(f"Permutation Importance: {name}")
        plt.xlabel("Decrease in F1/R2 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, f"{name}_permutation_importance.png"))
        plt.close()
        
        if task_type == "classification":
            # 3. ROC & PR Curves (using model decision values / probas)
            probas = model.predict_proba(X_val)
            preds = model.predict(X_val)
            
            # ROC curve for class 1 if binary classification
            if probas.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_val, probas[:, 1])
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve: {name}")
                plt.legend()
                plt.savefig(os.path.join(REPORTS_DIR, f"{name}_roc_curve.png"))
                plt.close()
                
                # PR Curve
                precision, recall, _ = precision_recall_curve(y_val, probas[:, 1])
                plt.figure(figsize=(6, 5))
                plt.plot(recall, precision, label="Precision-Recall Curve")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"PR Curve: {name}")
                plt.legend()
                plt.savefig(os.path.join(REPORTS_DIR, f"{name}_pr_curve.png"))
                plt.close()
                
                # Calibration curve
                prob_true, prob_pred = calibration_curve(y_val, probas[:, 1], n_bins=10)
                plt.figure(figsize=(6, 5))
                plt.plot(prob_pred, prob_true, marker='o', label="Calibration Curve")
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel("Predicted Probability")
                plt.ylabel("True Probability")
                plt.title(f"Calibration Curve: {name}")
                plt.legend()
                plt.savefig(os.path.join(REPORTS_DIR, f"{name}_calibration_curve.png"))
                plt.close()
        else:
            # 4. Residual plots for Regressor
            preds = model.predict(X_val)
            residuals = y_val - preds
            
            plt.figure(figsize=(8, 5))
            plt.scatter(preds, residuals, alpha=0.3)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel("Predictions")
            plt.ylabel("Residuals")
            plt.title(f"Residual Plot: {name}")
            plt.savefig(os.path.join(REPORTS_DIR, f"{name}_residual_plot.png"))
            plt.close()
    except Exception as e:
        print(f"  Explainability generation skipped for {name}: {e}")

# ── 5. MLOPS CHAMPION-CHALLENGER REGISTRY ───────────────────────────────────
def register_champion_challenger(name, version, task_type, framework, score_value, score_f1=None):
    try:
        Session = sessionmaker(bind=sync_engine)
        session = Session()
        
        # Query champion (active model) in database
        champion = session.query(ModelRegistry).filter_by(name=name, status="active").first()
        
        is_champion = True
        if champion:
            champ_metric = champion.accuracy if task_type == "classification" else champion.f1_score # using f1_score field for R2 score for regressor
            if score_value < champ_metric:
                is_champion = False
                print(f"  [MLOps] Challenger F1/R2 ({score_value:.4f}) did NOT beat Champion ({champ_metric:.4f}). Storing as staging.")
            else:
                print(f"  [MLOps] SUCCESS! Challenger F1/R2 ({score_value:.4f}) beat Champion ({champ_metric:.4f}). Promoting challenger!")
                champion.status = "archived"
                
        status = "active" if is_champion else "staging"
        
        reg = ModelRegistry(
            id=f"{name}_{version}_{int(time.time())}",
            name=name,
            version=version,
            type=task_type,
            framework=framework,
            status=status,
            accuracy=float(score_value) if task_type == "classification" else 0.0,
            f1_score=float(score_f1) if score_f1 is not None else float(score_value), # Store R2 in f1_score field for regressors
            last_retrained=datetime.utcnow()
        )
        session.add(reg)
        session.commit()
        session.close()
        return status
    except Exception as e:
        print(f"Warning: Champion-Challenger DB registry failed: {e}")
        return "active"

# ── 6. TUNING & BENCHMARKING ENGINE ──────────────────────────────────────────
def train_and_tune_model(X, y, task_type="classification", model_name="CatBoost", use_scaling=True, solve_imbalance=False):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if task_type=="classification" else None)
    
    if solve_imbalance and task_type == "classification":
        X_train, y_train, method = resolve_class_imbalance(X_train, y_train, X_val, y_val)
        
    if use_scaling:
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_train.columns)
    else:
        scaler = None
        X_train_s = X_train
        X_val_s = X_val
        
    if task_type == "classification":
        le_y = LabelEncoder()
        y_train_enc = le_y.fit_transform(y_train)
        y_val_enc = le_y.transform(y_val)
    else:
        y_train_enc = y_train.values
        y_val_enc = y_val.values
        le_y = None

    def build_estimator(m_name, params):
        if task_type == "classification":
            if m_name == "CatBoost":
                gpu_params = {**params, 'task_type': 'GPU', 'gpu_ram_part': 0.1, 'border_count': 32, 'used_ram_limit': '1Gb'}
                return CatBoostClassifier(**gpu_params)
            elif m_name == "LightGBM":
                return lgb.LGBMClassifier(**params, device='gpu')
            elif m_name == "XGBoost":
                return xgb.XGBClassifier(**params, device='cuda')
        else: # Regression
            if m_name == "CatBoost":
                gpu_params = {**params, 'task_type': 'GPU', 'gpu_ram_part': 0.1, 'border_count': 32, 'used_ram_limit': '1Gb'}
                return CatBoostRegressor(**gpu_params)
            elif m_name == "XGBoost":
                return xgb.XGBRegressor(**params, device='cuda')
            elif m_name == "LightGBM":
                return lgb.LGBMRegressor(**params, device='gpu')
        return None

    # Benchmark objective
    def objective(trial):
        if task_type == "classification":
            params = {
                'iterations': trial.suggest_int('iterations', 100, 150),
                'depth': trial.suggest_int('depth', 4, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'verbose': 0,
                'random_seed': 42
            } if model_name == "CatBoost" else {
                'n_estimators': trial.suggest_int('n_estimators', 100, 150),
                'max_depth': trial.suggest_int('max_depth', 4, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1
            }
            clf = build_estimator(model_name, params)
            if model_name == "CatBoost":
                clf.fit(X_train_s, y_train_enc, eval_set=(X_val_s, y_val_enc), early_stopping_rounds=15, verbose=0)
            else:
                clf.fit(X_train_s, y_train_enc, eval_set=[(X_val_s, y_val_enc)], callbacks=[lgb.early_stopping(15, verbose=False)])
            preds = clf.predict(X_val_s)
            return f1_score(y_val_enc, preds, average='macro')
        else: # Regression
            params = {
                'iterations': trial.suggest_int('iterations', 100, 150),
                'depth': trial.suggest_int('depth', 4, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'verbose': 0,
                'random_seed': 42
            } if model_name == "CatBoost" else {
                'n_estimators': trial.suggest_int('n_estimators', 100, 150),
                'max_depth': trial.suggest_int('max_depth', 4, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'random_state': 42,
                'n_jobs': -1
            }
            reg = build_estimator(model_name, params)
            if model_name == "CatBoost":
                reg.fit(X_train_s, y_train_enc, eval_set=(X_val_s, y_val_enc), early_stopping_rounds=20, verbose=0)
            elif model_name == "XGBoost":
                reg.fit(X_train_s, y_train_enc, eval_set=[(X_val_s, y_val_enc)], verbose=False)
            else:
                reg.fit(X_train_s, y_train_enc, eval_set=[(X_val_s, y_val_enc)])
            preds = reg.predict(X_val_s)
            return r2_score(y_val_enc, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3)
    best_params = study.best_params
    
    # Train final best estimator
    params = {**best_params, 'verbose': 0, 'random_seed': 42} if model_name=="CatBoost" else {**best_params, 'random_state':42, 'verbose': -1, 'n_jobs': -1}
    final_model = build_estimator(model_name, params)
    final_model.fit(X_train_s, y_train_enc)
    
    val_preds = final_model.predict(X_val_s)
    
    if task_type == "classification":
        score = f1_score(y_val_enc, val_preds, average='macro')
        acc = accuracy_score(y_val_enc, val_preds)
        print(f"  Benchmark {model_name} Classifier F1 Score: {score:.4f} (Accuracy: {acc:.4f})")
        generate_explainability_reports(final_model, X_val_s, y_val_enc, model_name, "classification")
        return final_model, scaler, le_y, score, acc
    else:
        score = r2_score(y_val_enc, val_preds)
        mae = mean_absolute_error(y_val_enc, val_preds)
        print(f"  Benchmark {model_name} Regressor R2 Score: {score:.4f} (MAE: {mae:.4f})")
        generate_explainability_reports(final_model, X_val_s, y_val_enc, model_name, "regression")
        return final_model, scaler, None, score, mae

# ── RUN MAIN ML PIPELINE ──────────────────────────────────────────────────────
def run_pipeline():
    print(f"Reading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Feature Engineering
    df = engineer_advanced_features(df)
    
    # Remove duplicate records
    df = df.drop_duplicates()
    metrics = {}
    
    # 2. CROP RECOMMENDATION (Voting Ensemble of Best benchmarks)
    # Target = Crop_Name
    crop_leaks = ['Crop_Category', 'Crop_Family', 'Variety', 'Hybrid', 'Expected_Yield', 'Actual_Yield', 
                  'Yield_per_Hectare', 'Marketable_Yield', 'Yield_Forecast', 'Profit', 'Net_Income', 'ROI', 'Harvest_Quality']
    X_crop, y_crop, _ = preprocess_and_select_features(df, 'Crop_Name', 'classification', crop_leaks)
    
    # Benchmark CatBoost, LightGBM, XGBoost
    cb_c, crop_sc, crop_le, cb_f1, cb_acc = train_and_tune_model(X_crop, y_crop, "classification", "CatBoost", use_scaling=False)
    lgb_c, _, _, lgb_f1, lgb_acc = train_and_tune_model(X_crop, y_crop, "classification", "LightGBM", use_scaling=False)
    
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y_crop)
    xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, device='cuda')
    xgb_c.fit(X_crop, y_encoded)
    xgb_f1 = f1_score(y_encoded, xgb_c.predict(X_crop), average='macro')
    
    # Ensemble weighted by relative F1 scores
    tot_f1 = cb_f1 + lgb_f1 + xgb_f1
    w_cb, w_lgb, w_xgb = cb_f1 / tot_f1, lgb_f1 / tot_f1, xgb_f1 / tot_f1
    print(f"  Ensemble weights: CatBoost={w_cb:.3f}, LightGBM={w_lgb:.3f}, XGBoost={w_xgb:.3f}")
    
    voting_ensemble = VotingClassifier(
        estimators=[('cb', cb_c), ('lgb', lgb_c), ('xgb', xgb_c)],
        voting='soft',
        weights=[w_cb, w_lgb, w_xgb]
    )
    voting_ensemble.fit(X_crop, y_encoded)
    
    joblib.dump(voting_ensemble, os.path.join(MODELS_DIR, "crop_recommendation_xgb.joblib"))
    joblib.dump(crop_le, os.path.join(MODELS_DIR, "crop_label_encoder.joblib"))
    joblib.dump(list(X_crop.columns), os.path.join(MODELS_DIR, "crop_feature_cols.joblib"))
    
    ensemble_acc = accuracy_score(y_encoded, voting_ensemble.predict(X_crop))
    ensemble_f1 = f1_score(y_encoded, voting_ensemble.predict(X_crop), average='macro')
    
    status = register_champion_challenger("Crop Recommendation Ensemble", "4.0.0", "classification", "weighted_ensemble", ensemble_acc, ensemble_f1)
    metrics["crop_recommendation"] = {"accuracy": ensemble_acc, "f1_score": ensemble_f1, "status": status}

    # 3. FERTILIZER RECOMMENDATION (CatBoost Classifier)
    print("\nSynthesizing Fertilizer Type target...")
    def map_fert(row):
        n, p, k = row['Nitrogen'], row['Phosphorus'], row['Potassium']
        if n > 55 and p < 25 and k < 25: return 'Urea'
        elif n > 45 and p > 45 and k < 25: return 'DAP'
        elif k > 45 and n < 25 and p < 25: return 'MOP'
        elif n > 20 and p > 20 and k > 20: return '10-26-26'
        elif n > 15 and p > 35 and k > 15: return '14-35-14'
        elif n > 20 and p > 20 and k < 15: return '20-20'
        elif n > 25 and p > 25 and k < 15: return '28-28'
        else: return 'NPK'
    df['Fertilizer_Type_Synth'] = df.apply(map_fert, axis=1)
    
    fert_leaks = ['Fertilizer_Type', 'Fertilizer_Grade', 'Fertilizer_Recommendation', 'Fertilizer_Cost', 
                  'Nitrogen_Applied', 'Phosphorus_Applied', 'Potassium_Applied']
    X_fert, y_fert, _ = preprocess_and_select_features(df, 'Fertilizer_Type_Synth', 'classification', fert_leaks)
    
    cb_fert, fert_scaler, fert_le, fert_f1, fert_acc = train_and_tune_model(X_fert, y_fert, "classification", "CatBoost", use_scaling=False, solve_imbalance=True)
    joblib.dump(cb_fert, os.path.join(MODELS_DIR, "fertilizer_recommendation_catboost.joblib"))
    joblib.dump(fert_le, os.path.join(MODELS_DIR, "fertilizer_label_encoder.joblib"))
    joblib.dump(list(X_fert.columns), os.path.join(MODELS_DIR, "fertilizer_feature_cols.joblib"))
    
    status = register_champion_challenger("Fertilizer Recommendation", "4.0.0", "classification", "catboost", fert_acc, fert_f1)
    metrics["fertilizer_recommendation"] = {"accuracy": fert_acc, "f1_score": fert_f1, "status": status}

    # 4. DISEASE PREDICTION (Ensemble + Calibrated Probabilities)
    # Target = Disease_Name
    dis_leaks = ['Disease_Class', 'Disease_Severity', 'Disease_Confidence', 'Infection_Rate', 
                 'Infection_Area', 'Leaf_Damage', 'Stem_Damage', 'Root_Damage', 'Disease_Risk_Index']
    X_dis, y_dis, _ = preprocess_and_select_features(df, 'Disease_Name', 'classification', dis_leaks)
    
    cb_dis, dis_scaler, dis_le, cb_dis_f1, cb_dis_acc = train_and_tune_model(X_dis, y_dis, "classification", "CatBoost", use_scaling=True, solve_imbalance=True)
    lgb_dis, _, _, lgb_dis_f1, lgb_dis_acc = train_and_tune_model(X_dis, y_dis, "classification", "LightGBM", use_scaling=True, solve_imbalance=True)
    
    # Create Calibrated probabilities
    y_dis_enc = dis_le.transform(y_dis)
    X_dis_scaled = dis_scaler.transform(X_dis)
    
    dis_ensemble = VotingClassifier(
        estimators=[('cb', cb_dis), ('lgb', lgb_dis)],
        voting='soft'
    )
    
    # Fit calibrated wrapper to optimize decision boundaries
    calibrated_ensemble = CalibratedClassifierCV(estimator=dis_ensemble, method='sigmoid', cv='prefit')
    
    # Fit ensemble then calibrate
    dis_ensemble.fit(X_dis_scaled, y_dis_enc)
    calibrated_ensemble.fit(X_dis_scaled, y_dis_enc)
    
    joblib.dump({
        'model': calibrated_ensemble,
        'scaler': dis_scaler,
        'feature_cols': list(X_dis.columns),
        'classes': list(dis_le.classes_)
    }, os.path.join(MODELS_DIR, "disease_risk_model.joblib"))
    
    dis_acc = accuracy_score(y_dis_enc, calibrated_ensemble.predict(X_dis_scaled))
    dis_f1 = f1_score(y_dis_enc, calibrated_ensemble.predict(X_dis_scaled), average='macro')
    
    status = register_champion_challenger("Disease Prediction Ensemble", "4.0.0", "classification", "calibrated_ensemble", dis_acc, dis_f1)
    metrics["disease_prediction"] = {"accuracy": dis_acc, "f1_score": dis_f1, "status": status}

    # 5. PEST PREDICTION (LightGBM)
    # Target = Pest_Name
    pest_leaks = ['Pest_Count', 'Pest_Density', 'Pest_Severity', 'Trap_Count', 'Infestation_Level', 'Pest_Risk_Index']
    X_pest, y_pest, _ = preprocess_and_select_features(df, 'Pest_Name', 'classification', pest_leaks)
    lgb_pest, pest_scaler, pest_le, pest_f1, pest_acc = train_and_tune_model(X_pest, y_pest, "classification", "LightGBM", use_scaling=True, solve_imbalance=True)
    
    joblib.dump({
        'model': lgb_pest,
        'scaler': pest_scaler,
        'label_encoder': pest_le,
        'feature_cols': list(X_pest.columns)
    }, os.path.join(MODELS_DIR, "pest_prediction_lightgbm.joblib"))
    
    status = register_champion_challenger("Pest Prediction", "4.0.0", "classification", "lightgbm", pest_acc, pest_f1)
    metrics["pest_prediction"] = {"accuracy": pest_acc, "f1_score": pest_f1, "status": status}

    # 6. WEED PREDICTION (CatBoost)
    # Target = Weed_Species
    weed_leaks = ['Weed_Density', 'Weed_Coverage', 'Weed_Height', 'Weed_Severity']
    X_weed, y_weed, _ = preprocess_and_select_features(df, 'Weed_Species', 'classification', weed_leaks)
    cb_weed, weed_scaler, weed_le, weed_f1, weed_acc = train_and_tune_model(X_weed, y_weed, "classification", "CatBoost", use_scaling=True, solve_imbalance=True)
    
    joblib.dump({
        'model': cb_weed,
        'scaler': weed_scaler,
        'label_encoder': weed_le,
        'feature_cols': list(X_weed.columns)
    }, os.path.join(MODELS_DIR, "weed_prediction_catboost.joblib"))
    
    status = register_champion_challenger("Weed Prediction", "4.0.0", "classification", "catboost", weed_acc, weed_f1)
    metrics["weed_prediction"] = {"accuracy": weed_acc, "f1_score": weed_f1, "status": status}

    # 7. YIELD PREDICTION (Regressor Benchmarking)
    # Target = Actual_Yield
    yield_leaks = ['Expected_Yield', 'Historical_Yield', 'Yield_Forecast', 'Yield_per_Hectare', 
                   'Marketable_Yield', 'Profit', 'Net_Income', 'ROI', 'Fruit_Count', 'Fruit_Size', 'Grain_Weight']
    X_yield, y_yield, _ = preprocess_and_select_features(df, 'Actual_Yield', 'regression', yield_leaks)
    
    # Benchmark CatBoost vs XGBoost vs LightGBM Regressors
    cb_y_model, cb_y_sc, _, cb_y_r2, cb_y_mae = train_and_tune_model(X_yield, y_yield, "regression", "CatBoost", use_scaling=False)
    xgb_y_model, xgb_y_sc, _, xgb_y_r2, xgb_y_mae = train_and_tune_model(X_yield, y_yield, "regression", "XGBoost", use_scaling=False)
    lgb_y_model, lgb_y_sc, _, lgb_y_r2, lgb_y_mae = train_and_tune_model(X_yield, y_yield, "regression", "LightGBM", use_scaling=False)
    
    best_y_model = cb_y_model
    best_y_r2 = cb_y_r2
    best_y_mae = cb_y_mae
    best_y_framework = "CatBoostRegressor"
    
    if xgb_y_r2 > best_y_r2:
        best_y_model = xgb_y_model
        best_y_r2 = xgb_y_r2
        best_y_mae = xgb_y_mae
        best_y_framework = "XGBoostRegressor"
    if lgb_y_r2 > best_y_r2:
        best_y_model = lgb_y_model
        best_y_r2 = lgb_y_r2
        best_y_mae = lgb_y_mae
        best_y_framework = "LGBMRegressor"
        
    print(f"  Yield prediction best benchmark performer: {best_y_framework} (R2={best_y_r2:.4f})")
    
    crop_le_enc = LabelEncoder().fit(df['Crop_Name'].astype(str))
    season_le_enc = LabelEncoder().fit(df['Crop_Season'].astype(str))
    state_le_enc = LabelEncoder().fit(df['Variety'].astype(str))
    yield_encoders_dict = {"Crop": crop_le_enc, "Season": season_le_enc, "State": state_le_enc}
    
    joblib.dump(best_y_model, os.path.join(MODELS_DIR, "yield_prediction_catboost.joblib"))
    joblib.dump(yield_encoders_dict, os.path.join(MODELS_DIR, "yield_encoders.joblib"))
    joblib.dump(cb_y_sc, os.path.join(MODELS_DIR, "yield_scaler.joblib"))
    joblib.dump(list(X_yield.columns), os.path.join(MODELS_DIR, "yield_feature_cols.joblib"))
    
    status = register_champion_challenger("Yield Prediction", "4.0.0", "regression", best_y_framework, best_y_r2)
    metrics["yield_prediction"] = {"r2_score": best_y_r2, "mae": best_y_mae, "status": status}

    # 8. IRRIGATION OPTIMIZATION (Regressor Benchmarking)
    # Target = Water_Applied
    irr_leaks = ['Irrigation_Duration', 'Irrigation_Interval', 'Water_Use_Efficiency', 'Soil_Water_Balance', 
                 'Root_Zone_Moisture']
    X_irr, y_irr, _ = preprocess_and_select_features(df, 'Water_Applied', 'regression', irr_leaks)
    
    cb_i_model, cb_i_sc, _, cb_i_r2, cb_i_mae = train_and_tune_model(X_irr, y_irr, "regression", "CatBoost", use_scaling=True)
    lgb_i_model, lgb_i_sc, _, lgb_i_r2, lgb_i_mae = train_and_tune_model(X_irr, y_irr, "regression", "LightGBM", use_scaling=True)
    
    best_i_model = cb_i_model
    best_i_scaler = cb_i_sc
    best_i_r2 = cb_i_r2
    best_i_mae = cb_i_mae
    best_i_framework = "CatBoostRegressor"
    
    if lgb_i_r2 > best_i_r2:
        best_i_model = lgb_i_model
        best_i_scaler = lgb_i_sc
        best_i_r2 = lgb_i_r2
        best_i_mae = lgb_i_mae
        best_i_framework = "LGBMRegressor"
        
    print(f"  Irrigation optimization best benchmark performer: {best_i_framework} (R2={best_i_r2:.4f})")
    
    irrigation_bundle = {
        'lgb': best_i_model,
        'scaler': best_i_scaler,
        'feature_cols': list(X_irr.columns),
        'log_transform': False,
        'blend': [1.0, 0.0]
    }
    joblib.dump(irrigation_bundle, os.path.join(MODELS_DIR, "irrigation_rf.joblib"))
    joblib.dump(list(X_irr.columns), os.path.join(MODELS_DIR, "irrigation_feature_cols.joblib"))
    
    status = register_champion_challenger("Irrigation Optimization", "4.0.0", "regression", best_i_framework, best_i_r2)
    metrics["irrigation"] = {"r2_score": best_i_r2, "mae": best_i_mae, "status": status}

    # 9. DIGITAL TWIN (Hybrid Physics equations & residual learning)
    print("\nTraining Digital Twin Hybrid Residual Estimators...")
    
    # Physics/Agronomic equations baseline functions
    def physics_water_stress(row):
        moisture = row.get('Soil_Moisture_Surface', 35.0)
        temp = row.get('Air_Temperature', 25.0)
        base_stress = np.clip((45.0 - moisture) / 45.0, 0.0, 1.0)
        heat_factor = 1.0 + np.maximum(0.0, temp - 32.0) * 0.05
        return np.clip(base_stress * heat_factor, 0.0, 1.0)
        
    def physics_disease_risk(row):
        humidity = row.get('Humidity', 60.0)
        temp = row.get('Air_Temperature', 25.0)
        rainfall_24 = row.get('Rainfall_24h', 5.0)
        risk = (humidity / 100.0) * (1.0 - np.abs(temp - 23.0) / 10.0) * (1.0 + rainfall_24 / 10.0)
        return np.clip(risk, 0.0, 1.0)
        
    def physics_growth_index(row):
        temp = row.get('Air_Temperature', 25.0)
        moisture = row.get('Soil_Moisture_Surface', 35.0)
        age = row.get('Plant_Age', 30.0)
        gdd = np.maximum(0.0, temp - 10.0)
        index = (gdd / 15.0) * (moisture / 35.0) * (1.0 + age / 120.0)
        return np.clip(index, 0.0, 1.0)

    # Compute baseline and residual targets
    df['water_stress_phys'] = df.apply(physics_water_stress, axis=1)
    df['disease_risk_phys'] = df.apply(physics_disease_risk, axis=1)
    df['growth_sim_phys'] = df.apply(physics_growth_index, axis=1)
    
    # Calculate residuals
    df['water_stress_residual'] = df['Water_Stress_Index'] - df['water_stress_phys']
    df['disease_risk_residual'] = df['Disease_Risk_Index'] - df['disease_risk_phys']
    df['growth_sim_residual'] = df['Growth_Simulation_Index'] - df['growth_sim_phys']
    
    twin_targets = {
        'Water_Stress_Index': ('water_stress_residual', 'water_stress_phys', "water_stress"),
        'Disease_Risk_Index': ('disease_risk_residual', 'disease_risk_phys', "disease_risk"),
        'Growth_Simulation_Index': ('growth_sim_residual', 'growth_sim_phys', "growth_simulation")
    }
    
    for real_col, (res_col, phys_col, phys_type) in twin_targets.items():
        twin_leaks = list(twin_targets.keys()) + [res_col, phys_col, 'water_stress_phys', 'disease_risk_phys', 'growth_sim_phys']
        X_twin, y_twin, _ = preprocess_and_select_features(df, res_col, 'regression', twin_leaks)
        
        cb_res_model, twin_scaler, _, res_r2, res_mae = train_and_tune_model(X_twin, y_twin, "regression", "CatBoost", use_scaling=True)
        
        # Save bundle containing model and the physics equation type indicator string
        bundle_path = os.path.join(MODELS_DIR, f"twin_{real_col.lower()}_catboost.joblib")
        joblib.dump({
            'model': cb_res_model,
            'scaler': twin_scaler,
            'feature_cols': list(X_twin.columns),
            'physics_type': phys_type
        }, bundle_path)
        
        status = register_champion_challenger(f"Digital Twin Residual - {real_col}", "4.0.0", "regression", "catboost_residual", res_r2)
        metrics[f"twin_{real_col.lower()}"] = {"residual_r2": res_r2, "mae": res_mae, "status": status}

    # 10. DIGITAL TWIN - YIELD FORECAST (CatBoost Regressor)
    X_yf, y_yf, _ = preprocess_and_select_features(df, 'Yield_Forecast', 'regression', list(twin_targets.keys()))
    cb_yf, yf_scaler, _, yf_r2, yf_mae = train_and_tune_model(X_yf, y_yf, "regression", "CatBoost", use_scaling=True)
    
    joblib.dump({
        'model': cb_yf,
        'scaler': yf_scaler,
        'feature_cols': list(X_yf.columns)
    }, os.path.join(MODELS_DIR, "twin_yield_forecast_catboost.joblib"))
    
    status = register_champion_challenger("Digital Twin - Yield Forecast", "4.0.0", "regression", "catboost", yf_r2)
    metrics["twin_yield_forecast"] = {"r2_score": yf_r2, "mae": yf_mae, "status": status}

    # Save summary metrics.json
    metrics["last_trained"] = datetime.utcnow().isoformat() + "Z"
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nAll benchmarked models optimized! Metrics saved to {MODELS_DIR}/metrics.json")

if __name__ == "__main__":
    run_pipeline()
