import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Paths
DATA_PATH = os.path.join("..", "..", "indian_agriculture_ml_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def train_yield_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print(f"Loading dataset from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        # Try local path if called from root
        df = pd.read_csv("indian_agriculture_ml_dataset.csv")

    # Feature Selection
    features = [
        "soil_n",
        "soil_p",
        "soil_k",
        "soil_ph",
        "organic_carbon",
        "rainfall_mm",
        "temperature_avg_c",
        "humidity_pct",
        "crop_name",
        "season",
    ]
    target = "yield_kg"

    # Drop rows with missing values in target or features
    df = df.dropna(subset=features + [target])

    X = df[features].copy()
    y = df[target]

    # Encoding Categorical Features
    encoders = {}
    categorical_cols = ["crop_name", "season"]

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        print(f"Encoded {col}: {le.classes_}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost Regressor
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    # Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\nModel Evaluation:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.2f}")

    # Save Model and Encoders
    model_path = os.path.join(MODEL_DIR, "yield_model.json")
    encoders_path = os.path.join(MODEL_DIR, "yield_encoders.joblib")

    model.save_model(model_path)
    joblib.dump(encoders, encoders_path)

    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Encoders saved to: {encoders_path}")


if __name__ == "__main__":
    train_yield_model()
