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
# STAGE 3A: OPTUNA TUNING (GROUP)
# ----------------------------------------------------

BASE_DIR = Path("f:/AGRISENSEFULL-STACK/backend/ml")
DATA_PATH = BASE_DIR / "datasets_enhanced/crop_recommendation/crop_data_enhanced.csv"
MODEL_PATH = BASE_DIR / "models/audit_locked/crop_group_model.pkl"

CROP_MAP = {
    "Rice": "cereal",
    "Wheat": "cereal",
    "Maize": "cereal",
    "Barley": "cereal",
    "Oats": "cereal",
    "Buckwheat": "cereal",
    "Bajra": "millet",
    "Jowar": "millet",
    "Ragi": "millet",
    "Sorghum": "millet",
    "Barnyard_Millet": "millet",
    "Foxtail_Millet": "millet",
    "Kodo_Millet": "millet",
    "Little_Millet": "millet",
    "Pearl_Millet": "millet",
    "Proso_Millet": "millet",
    "Chickpea": "pulse",
    "Pigeon_Pea": "pulse",
    "Arhar": "pulse",
    "Moong": "pulse",
    "Masoor": "pulse",
    "Urad": "pulse",
    "Lentil": "pulse",
    "Kidney_Bean": "pulse",
    "Moth_Bean": "pulse",
    "Horse_Gram": "pulse",
    "Cluster_Bean": "pulse",
    "French_Bean": "pulse",
    "Green_Pea": "pulse",
    "Field_Pea": "pulse",
    "Soybean": "pulse",
    "Tomato": "vegetable",
    "Brinjal": "vegetable",
    "Chilli": "vegetable",
    "Okra": "vegetable",
    "Cabbage": "vegetable",
    "Cauliflower": "vegetable",
    "Spinach": "vegetable",
    "Lettuce": "vegetable",
    "Pumpkin": "vegetable",
    "Bottle_Gourd": "vegetable",
    "Bitter_Gourd": "vegetable",
    "Ridge_Gourd": "vegetable",
    "Cucumber": "vegetable",
    "Drumstick": "vegetable",
    "Potato": "tuber",
    "Onion": "tuber",
    "Garlic": "tuber",
    "Ginger": "tuber",
    "Turmeric": "tuber",
    "Radish": "tuber",
    "Carrot": "tuber",
    "Beetroot": "tuber",
    "Sweet_Potato": "tuber",
    "Turnip": "tuber",
    "Mango": "fruit",
    "Banana": "fruit",
    "Papaya": "fruit",
    "Guava": "fruit",
    "Pomegranate": "fruit",
    "Sapota": "fruit",
    "Jackfruit": "fruit",
    "Custard_Apple": "fruit",
    "Litchi": "fruit",
    "Pineapple": "fruit",
    "Dragon_Fruit": "fruit",
    "Watermelon": "fruit",
    "Muskmelon": "fruit",
    "Apple": "cold",
    "Orange": "cold",
    "Grapes": "cold",
    "Strawberry": "cold",
    "Walnut": "cold",
    "Almond": "cold",
    "Sugarcane": "plantation",
    "Cotton": "fiber",
    "Jute": "fiber",
    "Rubber": "plantation",
    "Tea": "plantation",
    "Coffee": "plantation",
    "Coconut": "plantation",
    "Arecanut": "plantation",
    "Cashew": "plantation",
    "Tobacco": "plantation",
    "Black_Pepper": "spice",
    "Cardamom": "spice",
    "Clove": "spice",
    "Cumin": "spice",
    "Coriander": "spice",
    "Fenugreek": "spice",
    "Groundnut": "oilseed",
    "Mustard": "oilseed",
    "Sunflower": "oilseed",
    "Sesame": "oilseed",
    "Castor": "oilseed",
    "Linseed": "oilseed",
    "Safflower": "oilseed",
    "Niger": "oilseed",
}


def objective(trial):
    df = pd.read_csv(DATA_PATH)
    df["crop_group"] = df["label"].map(CROP_MAP).fillna("vegetable")

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    le = LabelEncoder()
    df["target_enc"] = le.fit_transform(df["crop_group"])

    df["dummy_dist"] = np.arange(len(df)) % 10

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "objective": "multi:softprob",
        "random_state": 42,
    }

    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(df, groups=df["dummy_dist"]))
    X_train, X_test = df[features].iloc[train_idx], df[features].iloc[test_idx]
    y_train, y_test = df["target_enc"].iloc[train_idx], df["target_enc"].iloc[test_idx]

    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\nBest Accuracy:", study.best_value)

    if study.best_value >= 0.95:
        df = pd.read_csv(DATA_PATH)
        df["crop_group"] = df["label"].map(CROP_MAP).fillna("vegetable")
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        le = LabelEncoder()
        df["target_enc"] = le.fit_transform(df["crop_group"])

        model = xgb.XGBClassifier(**study.best_params)
        model.fit(df[features], df["target_enc"])

        joblib.dump(model, MODEL_PATH)
        joblib.dump(le, (BASE_DIR / "models/audit_locked/crop_group_encoder.pkl"))
        joblib.dump(
            features, (BASE_DIR / "models/audit_locked/crop_group_features.pkl")
        )
        print("💾 Optimized group model locked.")
