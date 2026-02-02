import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import optuna
import joblib
from pathlib import Path

# ----------------------------------------------------
# STAGE 2A: OPTUNA TUNING
# ----------------------------------------------------

BASE_DIR = Path("f:/AGRISENSEFULL-STACK/backend/ml")
DATA_PATH = (
    BASE_DIR / "datasets_enhanced/season_classification/season_data_enhanced.csv"
)
MODEL_PATH = BASE_DIR / "models/audit_locked/season_model.pkl"


def objective(trial):
    df = pd.read_csv(DATA_PATH)
    # Feature engineering for signal boost
    df["temp_hum"] = df["temperature"] * df["humidity"]
    df["is_monsoon"] = (df["rainfall"] > 150).astype(int)

    features = [
        "temperature",
        "humidity",
        "rainfall",
        "growth_duration",
        "temp_hum",
        "is_monsoon",
    ]
    target = "label"

    le = LabelEncoder()
    df["target_enc"] = le.fit_transform(df[target])

    df["dummy_dist"] = np.arange(len(df)) % 10

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "objective": "multi:softprob",
        "random_state": 42,
        "n_jobs": -1,
    }

    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(df, groups=df["dummy_dist"]))
    X_train, X_test = df[features].iloc[train_idx], df[features].iloc[test_idx]
    y_train, y_test = df["target_enc"].iloc[train_idx], df["target_enc"].iloc[test_idx]

    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\nBest Accuracy:", study.best_value)
    print("Best Params:", study.best_params)

    # Final train and lock
    if study.best_value >= 0.95:
        df = pd.read_csv(DATA_PATH)
        df["temp_hum"] = df["temperature"] * df["humidity"]
        df["is_monsoon"] = (df["rainfall"] > 150).astype(int)
        features = [
            "temperature",
            "humidity",
            "rainfall",
            "growth_duration",
            "temp_hum",
            "is_monsoon",
        ]
        le = LabelEncoder()
        df["target_enc"] = le.fit_transform(df["label"])

        model = xgb.XGBClassifier(**study.best_params)
        model.fit(df[features], df["target_enc"])

        joblib.dump(model, MODEL_PATH)
        joblib.dump(le, (BASE_DIR / "models/audit_locked/season_encoder.pkl"))
        joblib.dump(features, (BASE_DIR / "models/audit_locked/season_features.pkl"))
        print("💾 Optimized model saved.")
