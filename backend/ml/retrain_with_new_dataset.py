import json
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

print("🌾 AgriSense Advanced Model Retraining with New Dataset")
print("=" * 70)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "retraining_results"
RESULTS_DIR.mkdir(exist_ok=True)

results = {}

# Try multiple locations for dataset
DATASET_LOCATIONS = [
    Path.cwd() / "indian_agriculture_ml_dataset.csv",  # If run from root
    BASE_DIR.parent.parent
    / "indian_agriculture_ml_dataset.csv",  # If run from backend/ml
]

DATASET_PATH = None
for path in DATASET_LOCATIONS:
    if path.exists():
        DATASET_PATH = path
        break

if not DATASET_PATH:
    print(f"❌ Dataset not found. Checked: {[str(p) for p in DATASET_LOCATIONS]}")
    sys.exit(1)


def log_status(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_and_preprocess_data():
    log_status("📥 Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    log_status(f"   Loaded {len(df)} rows.")

    # Preprocessing
    # Map columns to standard names
    df_clean = df.rename(
        columns={
            "soil_n": "N",
            "soil_p": "P",
            "soil_k": "K",
            "soil_ph": "ph",
            "temperature_avg_c": "temperature",  # Prioritize avg
            "humidity_pct": "humidity",
            "rainfall_mm": "rainfall",
            "crop_name": "label",
            "water_required_m3": "water_need",
        }
    )

    # If temperature_avg_c is missing/null, try avg_temp_c
    if "temperature" not in df_clean.columns and "avg_temp_c" in df.columns:
        df_clean["temperature"] = df["avg_temp_c"]

    # Encode Soil Texture
    if "soil_texture" in df_clean.columns:
        texture_encoder = LabelEncoder()
        df_clean["soil_texture_encoded"] = texture_encoder.fit_transform(
            df_clean["soil_texture"].astype(str)
        )
        # Save encoder immediately or pass it out? passing it out is cleaner but complex refactor.
        # Let's save it to a global/artifact or return it.
        # For simplicity, we re-fit inside training or standardise here.
        # A better way: fit here, save mapping?
        # Let's just create a texture scaler/encoder in the training function.

    return df_clean


def train_crop_recommendation(df):
    log_status("🌾 Training Crop Recommendation Model...")

    try:
        # Prepare features list
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        target = "label"

        if "soil_texture" in df.columns:
            features.append("soil_texture")
        if "agro_climatic_zone" in df.columns:
            features.append("agro_climatic_zone")
        if "avg_rainfall_mm" in df.columns:
            features.append("avg_rainfall_mm")

        # Drop rows with missing values in features
        df_model = df[features + [target]].dropna()

        # Handle Soil Texture Encoding
        final_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        if "soil_texture" in df_model.columns:
            tex_enc = LabelEncoder()
            df_model["soil_texture_encoded"] = tex_enc.fit_transform(
                df_model["soil_texture"]
            )
            joblib.dump(tex_enc, MODELS_DIR / "soil_texture_encoder.pkl")
            final_features.append("soil_texture_encoded")

        # Encode Agro Climatic Zone
        if (
            "agro_climatic_zone" in df_model.columns
        ):  # Check df_model as it's already subsetted and dropped NA
            agro_enc = LabelEncoder()
            df_model["agro_climatic_zone_encoded"] = agro_enc.fit_transform(
                df_model["agro_climatic_zone"]
            )
            joblib.dump(agro_enc, MODELS_DIR / "agro_climatic_zone_encoder.pkl")
            final_features.append("agro_climatic_zone_encoded")

        if "avg_rainfall_mm" in df_model.columns:
            final_features.append("avg_rainfall_mm")

        X = df_model[final_features]
        y = df_model[target]

        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train with RandomForest (Parallel)
        log_status("   🚀 Training with RandomForestClassifier (Balanced)...")
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate Training Accuracy (To show model capacity)
        y_train_pred = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        log_status(f"   🔥 Training Accuracy: {train_acc:.4f}")

        # Evaluate Test
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        log_status(f"   ✅ Test Accuracy: {acc:.4f}")
        log_status(f"   ✅ Test F1 Score: {f1:.4f}")

        # Save
        joblib.dump(model, MODELS_DIR / "enhanced_crop_recommendation_model.pkl")
        joblib.dump(scaler, MODELS_DIR / "enhanced_crop_recommendation_scaler.pkl")
        joblib.dump(encoder, MODELS_DIR / "enhanced_crop_recommendation_encoder.pkl")

        log_status("   💾 Saved Crop Recommendation Model")

    except Exception as e:
        log_status(f"   ❌ Error training crop recommendation: {e}")


def train_yield_prediction(df):
    log_status("📈 Training Yield Prediction Model...")

    try:
        # Calculate yield hg/ha
        # 1 kg = 10 hg
        # 1 ha = 10000 m2
        # yield (hg/ha) = (yield_kg / area_m2) * 10 * 10000 = (yield_kg / area_m2) * 100000

        df["hg_ha_yield"] = (df["yield_kg"] / df["area_m2"]) * 100000

        # Let's use: N, P, K, temperature, humidity, rainfall, ph, Area(Ha), Crop
        df["Area_Ha"] = df["area_m2"] / 10000.0

        features = [
            "Area_Ha",
            "label",
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "rainfall",
        ]
        if "soil_texture" in df.columns:
            features.append("soil_texture")

        df_model = df[features + ["hg_ha_yield"]].dropna()

        # Encode Crop
        encoder = LabelEncoder()
        df_model["label_encoded"] = encoder.fit_transform(df_model["label"])

        base_features = [
            "Area_Ha",
            "label_encoded",
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "rainfall",
        ]

        # Add soil texture if available
        if "soil_texture" in df_model.columns:
            # Use the SAME encoder if possible, or fit new one?
            # Yield model should ideally reuse the same concept, but training separate encoder is safer for modularity.
            tex_enc_yield = LabelEncoder()
            df_model["soil_texture_encoded"] = tex_enc_yield.fit_transform(
                df_model["soil_texture"]
            )
            joblib.dump(tex_enc_yield, MODELS_DIR / "yield_soil_texture_encoder.pkl")
            base_features.append("soil_texture_encoded")

        X = df_model[base_features]
        y = df_model["hg_ha_yield"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = GradientBoostingRegressor(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        log_status(f"   ✅ R² Score: {r2:.4f}")
        log_status(f"   ✅ RMSE: {rmse:.2f}")

        joblib.dump(model, MODELS_DIR / "enhanced_yield_prediction_model.pkl")
        joblib.dump(scaler, MODELS_DIR / "enhanced_yield_prediction_scaler.pkl")
        # We need the encoder too if we are encoding items
        joblib.dump(encoder, MODELS_DIR / "enhanced_yield_prediction_encoder.pkl")

        log_status("   💾 Saved Yield Prediction Model")
        results["yield_prediction"] = {
            "r2": float(r2),
            "rmse": float(rmse),
            "samples": int(len(X_train)),
        }

    except Exception as e:
        log_status(f"   ❌ Error training yield prediction: {e}")


def train_water_requirement(df):
    log_status("💧 Training Water Requirement Model...")

    try:
        features = ["temperature", "humidity", "rainfall", "label"]
        target = "water_need"

        # Add soil texture
        if "soil_texture" in df.columns:
            features.append("soil_texture")

        df_model = df[features + [target]].dropna()

        # Encode Crop
        encoder = LabelEncoder()
        df_model["label_encoded"] = encoder.fit_transform(df_model["label"])

        train_features = ["temperature", "humidity", "rainfall", "label_encoded"]

        if "soil_texture" in df_model.columns:
            tex_enc_water = LabelEncoder()
            df_model["soil_texture_encoded"] = tex_enc_water.fit_transform(
                df_model["soil_texture"]
            )
            joblib.dump(tex_enc_water, MODELS_DIR / "water_soil_texture_encoder.pkl")
            train_features.append("soil_texture_encoded")

        X = df_model[train_features]
        y = df_model[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)

        log_status(f"   ✅ R² Score: {r2:.4f}")

        joblib.dump(model, MODELS_DIR / "enhanced_water_requirement_model.pkl")
        joblib.dump(scaler, MODELS_DIR / "enhanced_water_requirement_scaler.pkl")
        joblib.dump(encoder, MODELS_DIR / "enhanced_water_requirement_encoder.pkl")

        log_status("   💾 Saved Water Requirement Model")
        results["water_requirement"] = {
            "r2": float(r2),
            "samples": int(len(X_train)),
        }

    except Exception as e:
        log_status(f"   ❌ Error training water requirement: {e}")


def main():
    df = load_and_preprocess_data()
    train_crop_recommendation(df)
    train_yield_prediction(df)
    train_water_requirement(df)
    # --- Additional classifiers ---
    try:
        # Crop Type Classification
        if "crop_type" in df.columns:
            log_status("🌱 Training Crop Type Classification...")
            from sklearn.ensemble import RandomForestClassifier

            features = [
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall",
                "ndvi_mean",
            ]
            cols = [c for c in features if c in df.columns]
            df_model = df[cols + ["crop_type"]].dropna()
            if len(df_model) > 50:
                enc = LabelEncoder()
                y = enc.fit_transform(df_model["crop_type"])
                X = df_model[cols]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                clf = RandomForestClassifier(
                    n_estimators=150, random_state=42, n_jobs=-1
                )
                clf.fit(X_train_s, y_train)
                ypred = clf.predict(X_test_s)
                acc = accuracy_score(y_test, ypred)
                f1 = f1_score(y_test, ypred, average="weighted")
                joblib.dump(clf, MODELS_DIR / "crop_type_classifier.pkl")
                joblib.dump(scaler, MODELS_DIR / "crop_type_scaler.pkl")
                joblib.dump(enc, MODELS_DIR / "crop_type_encoder.pkl")
                results["crop_type_classification"] = {
                    "accuracy": float(acc),
                    "f1": float(f1),
                    "samples": int(len(X_train)),
                }
                log_status(f"   ✅ Crop type acc {acc:.4f} f1 {f1:.4f}")

    except Exception as e:
        log_status(f"   ❌ Error training crop type: {e}")

    try:
        # Season classification
        if "season" in df.columns:
            log_status("☀️ Training Season Classification...")
            features = ["N", "P", "K", "temperature", "rainfall", "ndvi_mean"]
            cols = [c for c in features if c in df.columns]
            df_model = df[cols + ["season"]].dropna()
            if len(df_model) > 50:
                enc = LabelEncoder()
                y = enc.fit_transform(df_model["season"])
                X = df_model[cols]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                clf = RandomForestClassifier(
                    n_estimators=150, random_state=42, n_jobs=-1
                )
                clf.fit(X_train_s, y_train)
                ypred = clf.predict(X_test_s)
                acc = accuracy_score(y_test, ypred)
                f1 = f1_score(y_test, ypred, average="weighted")
                joblib.dump(clf, MODELS_DIR / "season_classifier.pkl")
                joblib.dump(scaler, MODELS_DIR / "season_scaler.pkl")
                joblib.dump(enc, MODELS_DIR / "season_encoder.pkl")
                results["season_classification"] = {
                    "accuracy": float(acc),
                    "f1": float(f1),
                    "samples": int(len(X_train)),
                }
                log_status(f"   ✅ Season acc {acc:.4f} f1 {f1:.4f}")
    except Exception as e:
        log_status(f"   ❌ Error training season classifier: {e}")

    # Save retraining results
    try:
        out = (
            RESULTS_DIR
            / f'retraining_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(out, "w") as fh:
            json.dump(
                {"timestamp": datetime.now().isoformat(), "results": results},
                fh,
                indent=2,
            )
        log_status(f"💾 Retraining results saved to {out}")
    except Exception as e:
        log_status(f"❌ Failed to save retraining results: {e}")
    log_status("🏁 Retraining Complete.")


if __name__ == "__main__":
    main()
