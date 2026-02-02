import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# ----------------------------------------------------
# STAGE 3A: CROP GROUP CLASSIFICATION
# ----------------------------------------------------

BASE_DIR = Path("f:/AGRISENSEFULL-STACK/backend/ml")
DATA_PATH = BASE_DIR / "datasets_enhanced/crop_recommendation/crop_data_enhanced.csv"
MODEL_PATH = BASE_DIR / "models/audit_locked/crop_group_model.pkl"

CATEGORIES = {
    # Cereals
    "Rice": "Cereal",
    "Maize": "Cereal",
    "Wheat": "Cereal",
    "Bajra": "Cereal",
    "Barley": "Cereal",
    "Jowar": "Cereal",
    "Ragi": "Cereal",
    "Sorghum": "Cereal",
    "Millet": "Cereal",
    # Pulses
    "Arhar": "Pulse",
    "Chickpea": "Pulse",
    "GreenGram": "Pulse",
    "BlackGram": "Pulse",
    "Lentil": "Pulse",
    "Pea": "Pulse",
    "KidneyBeans": "Pulse",
    "MothBeans": "Pulse",
    "Cowpea": "Pulse",
    # Fruits
    "Apple": "Fruit",
    "Banana": "Fruit",
    "Orange": "Fruit",
    "Mango": "Fruit",
    "Grapes": "Fruit",
    "Pomegranate": "Fruit",
    "Papaya": "Fruit",
    "Watermelon": "Fruit",
    "Muskmelon": "Fruit",
    "Pineapple": "Fruit",
    "Guava": "Fruit",
    "Sapota": "Fruit",
    "Strawberry": "Fruit",
    # Vegetables
    "Potato": "Vegetable",
    "Tomato": "Vegetable",
    "Onion": "Vegetable",
    "Cabbage": "Vegetable",
    "Cauliflower": "Vegetable",
    "Carrot": "Vegetable",
    "Radish": "Vegetable",
    "Spinach": "Vegetable",
    "Brinjal": "Vegetable",
    "Okra": "Vegetable",
    "Chilli": "Vegetable",
    "Garlic": "Vegetable",
    "Ginger": "Vegetable",
    # Spices
    "BlackPepper": "Spice",
    "Cardamom": "Spice",
    "Clove": "Spice",
    "Cinnamon": "Spice",
    "Cumin": "Spice",
    "Turmeric": "Spice",
    "Coriander": "Spice",
    "Fennel": "Spice",
    "Fenugreek": "Spice",
    # Oilseeds
    "Groundnut": "Oilseed",
    "Sunflower": "Oilseed",
    "Mustard": "Oilseed",
    "Soybean": "Oilseed",
    "Sesame": "Oilseed",
    "Castor": "Oilseed",
    "Linseed": "Oilseed",
    "Safflower": "Oilseed",
    # Cash Crops
    "Cotton": "Cash",
    "Jute": "Cash",
    "Sugarcane": "Cash",
    "Tobacco": "Cash",
    "Tea": "Cash",
    "Coffee": "Cash",
    "Rubber": "Cash",
    "Coconut": "Cash",
    "Arecanut": "Cash",
    "Cashew": "Cash",
    # Medicinal
    "AloeVera": "Medicinal",
    "Ashwagandha": "Medicinal",
    "Neem": "Medicinal",
    "Tulsi": "Medicinal",
    "Stevia": "Medicinal",
    "Mentha": "Medicinal",
    "Senna": "Medicinal",
    # Flowers
    "Rose": "Flower",
    "Marigold": "Flower",
    "Jasmine": "Flower",
    "Hibiscus": "Flower",
    "Sunflower": "Flower",
    "Orchid": "Flower",
    "Chrysanthemum": "Flower",
}


def train_group_model():
    print("🚀 Starting Stage 3A: Crop Group Classification Training")
    df = pd.read_csv(DATA_PATH)

    # 1. Map Labels to Groups
    df["crop_group"] = df["label"].map(CATEGORIES).fillna("Other")

    # 2. Prep
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    le = LabelEncoder()
    df["target_enc"] = le.fit_transform(df["crop_group"])

    # dummy districts for split
    df["dummy_dist"] = np.arange(len(df)) % 10

    # 3. Spatial Split
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(df, groups=df["dummy_dist"]))

    X_train, X_test = df[features].iloc[train_idx], df[features].iloc[test_idx]
    y_train, y_test = df["target_enc"].iloc[train_idx], df["target_enc"].iloc[test_idx]

    # 4. Train
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        objective="multi:softprob",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ Results for Stage 3A:")
    print(f"Accuracy: {acc:.4f}")

    if acc >= 0.95:
        print(f"🏆 SUCCESS: Accuracy target achieved.")
        joblib.dump(model, MODEL_PATH)
        joblib.dump(le, (BASE_DIR / "models/audit_locked/crop_group_encoder.pkl"))
        print(f"💾 Model and encoder locked.")
    else:
        print(f"❌ FAILURE: Accuracy below target.")

    return acc


if __name__ == "__main__":
    train_group_model()
