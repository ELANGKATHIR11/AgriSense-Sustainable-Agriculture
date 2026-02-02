import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from pathlib import Path

# ----------------------------------------------------
# STAGE 1A: WATER REQUIREMENT MODEL (PHYSICS ANCHORS)
# ----------------------------------------------------

BASE_DIR = Path("f:/AGRISENSEFULL-STACK/backend/ml")
DATA_PATH = (
    BASE_DIR / "datasets_enhanced/water_requirement/water_requirement_enhanced.csv"
)
MODEL_PATH = BASE_DIR / "models/audit_locked/water_model.pkl"
(BASE_DIR / "models/audit_locked").mkdir(parents=True, exist_ok=True)

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


def train_water_model():
    print("🚀 Training Stage 1A with Physics Anchors")
    df = pd.read_csv(DATA_PATH)

    # 1. Physical Component Features
    def get_kc(crop):
        if crop in CROP_MAP:
            return CROP_MAP[crop]
        cat = CATEGORIES.get(crop, "vegetable")
        return KC_VALUES.get(cat, 0.9)

    df["eto"] = (
        (0.0023 * df["temperature"] + 0.3)
        * (0.6 + 0.4 * (1 - df["humidity"] / 100))
        * 5
    )
    df["kc"] = df["crop"].apply(get_kc)
    df["etc"] = df["eto"] * df["kc"]
    df["rain_eff"] = df["rainfall"] * 0.8
    # The "Anchor" is the pure physics formula
    df["physics_anchor"] = (df["etc"] - df["rain_eff"]).clip(0)

    features = ["physics_anchor", "eto", "kc", "rainfall", "area"]
    target = "water_requirement"

    # 2. Strict Row-wise Holdout
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )

    # 3. Model
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.01,
        monotone_constraints=(1, 1, 1, -1, 1),
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\n✅ Stage 1A R² Score: {r2:.6f}")

    if r2 >= 0.97:
        print("🏆 SUCCESS")
        joblib.dump(model, MODEL_PATH)
        joblib.dump(features, (BASE_DIR / "models/audit_locked/water_features.pkl"))
    return r2


if __name__ == "__main__":
    train_water_model()
