import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import joblib
from pathlib import Path
import optuna

# ----------------------------------------------------
# STAGE 1A: OPTUNA TUNING
# ----------------------------------------------------

BASE_DIR = Path("f:/AGRISENSEFULL-STACK/backend/ml")
DATA_PATH = (
    BASE_DIR / "datasets_enhanced/water_requirement/water_requirement_enhanced.csv"
)
MODEL_PATH = BASE_DIR / "models/audit_locked/water_model.pkl"


def get_data():
    df = pd.read_csv(DATA_PATH)
    # Categorical and mapping
    CROP_MAP = {"Rice": 1.25, "Sugarcane": 1.25, "Watermelon": 1.0}
    CATEGORIES = {
        "Almond": "cold",
        "Apple": "cold",
        "Arecanut": "plantation",
        "Arhar": "pulse",
        "Bajra": "millet",
        "Banana": "fruit",
        "Barley": "cereal",
        "Barnyard_Millet": "millet",
        "Beetroot": "tuber",
        "Bitter_Gourd": "vegetable",
        "Black_Pepper": "spice",
        "Bottle_Gourd": "vegetable",
        "Brinjal": "vegetable",
        "Buckwheat": "cereal",
        "Cabbage": "vegetable",
        "Cardamom": "spice",
        "Carrot": "tuber",
        "Cashew": "plantation",
        "Castor": "oilseed",
        "Cauliflower": "vegetable",
        "Chickpea": "pulse",
        "Chilli": "vegetable",
        "Cluster_Bean": "pulse",
        "Coconut": "plantation",
        "Coffee": "plantation",
        "Coriander": "spice",
        "Cotton": "fiber",
        "Cucumber": "vegetable",
        "Cumin": "spice",
        "Custard_Apple": "fruit",
        "Dragon_Fruit": "fruit",
        "Fenugreek": "spice",
        "Field_Pea": "pulse",
        "Foxtail_Millet": "millet",
        "French_Bean": "pulse",
        "Garlic": "tuber",
        "Ginger": "tuber",
        "Grapes": "cold",
        "Green_Pea": "pulse",
        "Groundnut": "oilseed",
        "Guava": "fruit",
        "Horse_Gram": "pulse",
        "Jackfruit": "fruit",
        "Jowar": "millet",
        "Jute": "fiber",
        "Kidney_Bean": "pulse",
        "Kodo_Millet": "millet",
        "Lentil": "pulse",
        "Lettuce": "vegetable",
        "Linseed": "oilseed",
        "Litchi": "fruit",
        "Little_Millet": "millet",
        "Maize": "cereal",
        "Mango": "fruit",
        "Masoor": "pulse",
        "Moong": "pulse",
        "Moth_Bean": "pulse",
        "Muskmelon": "fruit",
        "Mustard": "oilseed",
        "Niger": "oilseed",
        "Oats": "cereal",
        "Okra": "vegetable",
        "Onion": "tuber",
        "Orange": "cold",
        "Papaya": "fruit",
        "Pearl_Millet": "millet",
        "Pigeon_Pea": "pulse",
        "Pineapple": "fruit",
        "Pomegranate": "fruit",
        "Potato": "tuber",
        "Proso_Millet": "millet",
        "Pumpkin": "vegetable",
        "Radish": "tuber",
        "Ragi": "millet",
        "Rice": "cereal",
        "Ridge_Gourd": "vegetable",
        "Rubber": "plantation",
        "Safflower": "oilseed",
        "Sapota": "fruit",
        "Sesame": "oilseed",
        "Sorghum": "millet",
        "Soybean": "pulse",
        "Spinach": "vegetable",
        "Strawberry": "cold",
        "Sugarcane": "plantation",
        "Sunflower": "oilseed",
        "Sweet_Potato": "tuber",
        "Tea": "plantation",
        "Tobacco": "plantation",
        "Tomato": "vegetable",
        "Turmeric": "tuber",
        "Turnip": "tuber",
        "Urad": "pulse",
        "Walnut": "cold",
        "Watermelon": "fruit",
        "Wheat": "cereal",
    }
    KC_VALUES = {
        "cereal": 1.05,
        "millet": 0.7,
        "pulse": 0.75,
        "vegetable": 0.9,
        "fruit": 1.1,
        "plantation": 1.2,
        "fiber": 1.1,
        "sugar": 1.2,
        "spice": 0.85,
        "oilseed": 0.85,
        "tuber": 0.9,
        "cold": 1.1,
    }
    df["eto_proxy"] = (
        (0.0023 * df["temperature"] + 0.3)
        * (0.6 + 0.4 * (1 - df["humidity"] / 100))
        * 5
    )

    def get_kc(crop):
        if crop in CROP_MAP:
            return CROP_MAP[crop]
        cat = CATEGORIES.get(crop, "vegetable")
        return KC_VALUES.get(cat, 0.9)

    df["kc_value"] = df["crop"].apply(get_kc)
    df["etc_proxy"] = df["eto_proxy"] * df["kc_value"]

    unique_crops = df["crop"].unique()
    districts = [f"DIST_{i:03d}" for i in range(1, 11)]
    crop_to_dist = {crop: districts[i % 10] for i, crop in enumerate(unique_crops)}
    df["district_code"] = df["crop"].map(crop_to_dist)

    features = ["etc_proxy", "rainfall", "area"]
    return df, features, "water_requirement"


def objective(trial):
    df, features, target = get_data()
    constraints = (1, -1, 1)

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "monotone_constraints": constraints,
        "random_state": 42,
        "n_jobs": -1,
    }

    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(df, groups=df["district_code"]))
    X_train, X_test = df[features].iloc[train_idx], df[features].iloc[test_idx]
    y_train, y_test = df[target].iloc[train_idx], df[target].iloc[test_idx]

    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\nBest R2:", study.best_value)
    print("Best Params:", study.best_params)

    # Train final model
    df, features, target = get_data()
    constraints = (1, -1, 1)
    best_params = study.best_params
    best_params["monotone_constraints"] = constraints
    best_params["random_state"] = 42

    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(df, groups=df["district_code"]))
    X_train, X_test = df[features].iloc[train_idx], df[features].iloc[test_idx]
    y_train, y_test = df[target].iloc[train_idx], df[target].iloc[test_idx]

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"💾 Optimized model saved with R2: {study.best_value:.4f}")
