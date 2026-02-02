import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# ----------------------------------------------------
# STAGE 4: YIELD PREDICTION (96 CROPS)
# ----------------------------------------------------

BASE_DIR = Path("f:/AGRISENSEFULL-STACK/backend/ml")
DATA_PATH = BASE_DIR / "datasets_enhanced/yield_prediction/yield_data_enhanced.csv"
MODEL_PATH = BASE_DIR / "models/audit_locked/yield_model.pkl"


def train_yield_model():
    print("🚀 Starting Stage 4: Yield Prediction Training")
    df = pd.read_csv(DATA_PATH)

    # 1. Prep
    le = LabelEncoder()
    df["crop_enc"] = le.fit_transform(df["crop"])

    features = [
        "crop_enc",
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "rainfall",
        "area",
    ]
    target = "production"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Physics Constraints (Area must increase production)
    # feature_names: crop_enc(0), N(1), P(2), K(3), T(4), H(5), R(6), Area(7)
    monotone_constraints = (0, 0, 0, 0, 0, 0, 0, 1)

    # 3. XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.05,
        monotone_constraints=monotone_constraints,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n✅ Results for Stage 4:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")

    if r2 >= 0.90:
        print(f"🏆 SUCCESS: R² target achieves (>= 0.90).")
        joblib.dump(model, MODEL_PATH)
        joblib.dump(le, (BASE_DIR / "models/audit_locked/yield_encoder.pkl"))
        joblib.dump(features, (BASE_DIR / "models/audit_locked/yield_features.pkl"))
        print(f"💾 Model locked.")
    else:
        print(f"❌ FAILURE: R² below target.")

    return r2


if __name__ == "__main__":
    train_yield_model()
