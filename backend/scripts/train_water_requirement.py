import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import math

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT.parent / "AGRISENSEFULL-STACK" / "datasets" / "data" / "processed" / "water_requirement" / "water_requirement_train.csv"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_JSON = MODEL_DIR / "water_requirement_model.json"


def main():
    print("Loading dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    # Derive features to match frontend inputs
    df_features = pd.DataFrame({
        "temperature": (df["min_temp_C"] + df["max_temp_C"]) / 2,
        "humidity": df["moisture_max_percent"],  # proxy
        "rainfall": df["avg_rainfall"],
        "growth_duration": df["growth_duration_days"],
    })
    target_col = "target" if "target" in df.columns else df.columns[-1]
    y = df[target_col].values
    X = df_features.values
    feature_cols = df_features.columns

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror"
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    # Distill a linear surrogate for JS inference
    surrogate = LinearRegression()
    surrogate.fit(X_train_scaled, model.predict(X_train_scaled))
    coef_surrogate = surrogate.coef_.tolist()
    intercept_surrogate = float(surrogate.intercept_)

    print(f"R2: {r2:.4f}  RMSE: {rmse:.4f}")

    # Persist parameters for the JS service
    params_raw = model.get_params()
    params_clean = {}
    for k, v in params_raw.items():
        if isinstance(v, float) and math.isnan(v):
            params_clean[k] = None
        else:
            params_clean[k] = v

    payload = {
        "model": "XGBoostRegressor",
        "params": params_clean,
        "feature_importances": model.feature_importances_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "r2": float(r2),
        "rmse": float(rmse),
        "feature_names": feature_cols.tolist(),
        "target_name": target_col,
        "notes": "XGBoostRegressor with StandardScaler trained on derived 4-feature set aligned to frontend inputs",
        "model_file": str(MODEL_DIR / "water_requirement_xgb.json"),
        "surrogate": {
            "type": "LinearRegression",
            "coef": coef_surrogate,
            "intercept": intercept_surrogate
        }
    }

    MODEL_JSON.write_text(json.dumps(payload, indent=2))
    # Save booster
    model.save_model(payload["model_file"])
    print("Saved model parameters to", MODEL_JSON)
    print("Saved model booster to", payload["model_file"])


if __name__ == "__main__":
    main()
