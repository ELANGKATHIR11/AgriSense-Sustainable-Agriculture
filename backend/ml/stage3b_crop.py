import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# ----------------------------------------------------
# STAGE 3B: SPECIFIC CROP (HIERARCHICAL) - FIX
# ----------------------------------------------------

BASE_DIR = Path("f:/AGRISENSEFULL-STACK/backend/ml")
DATA_PATH = BASE_DIR / "soil_health_dataset.csv"
MODEL_DIR = BASE_DIR / "models/audit_locked/submodels"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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


def train_submodels():
    df = pd.read_csv(DATA_PATH)
    df["crop_group"] = df["label"].map(CROP_MAP).fillna("vegetable")
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    groups = df["crop_group"].unique()

    for group in groups:
        print(f"\n📂 Training Group: {group}")
        sub_df = df[df["crop_group"] == group].copy()

        n_classes = sub_df["label"].nunique()
        if n_classes < 2:
            print(
                f"   ⚠️ Skipping XGBoost: Only {n_classes} class ({sub_df['label'].iloc[0]})"
            )
            joblib.dump(sub_df["label"].iloc[0], MODEL_DIR / f"{group}_constant.pkl")
            continue

        le = LabelEncoder()
        sub_df["target_enc"] = le.fit_transform(sub_df["label"])

        sub_df["dummy_dist"] = np.arange(len(sub_df)) % 10
        gkf = GroupKFold(n_splits=5)
        train_idx, test_idx = next(gkf.split(sub_df, groups=sub_df["dummy_dist"]))

        X_train, X_test = (
            sub_df[features].iloc[train_idx],
            sub_df[features].iloc[test_idx],
        )
        y_train, y_test = (
            sub_df["target_enc"].iloc[train_idx],
            sub_df["target_enc"].iloc[test_idx],
        )

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        print(f"   Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

        model.fit(sub_df[features], sub_df["target_enc"])
        joblib.dump(model, MODEL_DIR / f"{group}_model.pkl")
        joblib.dump(le, MODEL_DIR / f"{group}_encoder.pkl")

    print("\n✅ End-to-End Hierarchical Species Models Locked.")


if __name__ == "__main__":
    train_submodels()
