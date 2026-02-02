import json
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------
# UNIFIED AGRI-INTELLIGENCE INFERENCE (STAGES 1-4)
# ----------------------------------------------------

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models/audit_locked"


def load_pkl(name):
    path = MODEL_DIR / name
    if path.exists():
        return joblib.load(path)
    return None


def predict():
    try:
        # 1. Read Input
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"error": "No input received"}))
            return

        data = json.loads(input_data)

        # 2. Extract Features
        # Features needed: N, P, K, temperature, humidity, ph, rainfall, area
        N = float(data.get("N", 0))
        P = float(data.get("P", 0))
        K = float(data.get("K", 0))
        temp = float(data.get("temperature", 25))
        hum = float(data.get("humidity", 60))
        ph = float(data.get("ph", 6.5))
        rain = float(data.get("rainfall", 100))
        area = float(data.get("area", 1.0))

        results = {}

        # --- STAGE 1: WATER REQUIREMENT ---
        water_model = load_pkl("water_model.pkl")
        if water_model:
            # Physics calculation from enhance_datasets.py
            eto = (0.0023 * temp + 0.3) * (0.6 + 0.4 * (1 - hum / 100)) * 5
            kc = 0.9  # Default for most crops in our training set
            physics_anchor = eto * kc

            # Expected Features: ['physics_anchor', 'eto', 'kc', 'rainfall', 'area']
            X_water = pd.DataFrame(
                [[physics_anchor, eto, kc, rain, area]],
                columns=["physics_anchor", "eto", "kc", "rainfall", "area"],
            )
            results["water_requirement"] = float(water_model.predict(X_water)[0])

        # --- STAGE 2: SEASON CLASSIFICATION ---
        season_model = load_pkl("season_model.pkl")
        season_le = load_pkl("season_encoder.pkl")
        if season_model and season_le:
            # Features: temperature, humidity, rainfall, growth_duration
            X_season = pd.DataFrame(
                [[temp, hum, rain, 120]],
                columns=["temperature", "humidity", "rainfall", "growth_duration"],
            )
            season_idx = season_model.predict(X_season)[0]
            results["season"] = season_le.inverse_transform([season_idx])[0]

        # --- STAGE 3A: CROP GROUPING ---
        group_model = load_pkl("crop_group_model.pkl")
        group_le = load_pkl("crop_group_encoder.pkl")
        if group_model and group_le:
            # Features: N, P, K, temperature, humidity, ph, rainfall
            X_group = pd.DataFrame(
                [[N, P, K, temp, hum, ph, rain]],
                columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
            )
            group_idx = group_model.predict(X_group)[0]
            group_name = group_le.inverse_transform([group_idx])[0]
            results["crop_group"] = group_name

            # --- STAGE 3B: SPECIFIC CROP (Within Group) ---
            sub_model = load_pkl(f"submodels/{group_name}_model.pkl")
            sub_le = load_pkl(f"submodels/{group_name}_encoder.pkl")
            if sub_model and sub_le:
                crop_idx = sub_model.predict(X_group)[0]
                crop_name = sub_le.inverse_transform([crop_idx])[0]
                results["recommended_crop"] = crop_name
            else:
                # Handle constant models (groups with 1 crop)
                constant_crop = load_pkl(f"submodels/{group_name}_constant.pkl")
                if constant_crop:
                    results["recommended_crop"] = constant_crop
                    crop_name = constant_crop
                else:
                    results["recommended_crop"] = "Unknown"
                    crop_name = "Unknown"

            # --- STAGE 4: YIELD PREDICTION ---
            yield_model = load_pkl("yield_model.pkl")
            yield_le = load_pkl("yield_encoder.pkl")
            if yield_model and yield_le and crop_name != "Unknown":
                try:
                    crop_enc = yield_le.transform([crop_name])[0]
                    # Features: ['crop_enc', 'N', 'P', 'K', 'temperature', 'humidity', 'rainfall', 'area']
                    X_yield = pd.DataFrame(
                        [[crop_enc, N, P, K, temp, hum, rain, area]],
                        columns=[
                            "crop_enc",
                            "N",
                            "P",
                            "K",
                            "temperature",
                            "humidity",
                            "rainfall",
                            "area",
                        ],
                    )
                    results["expected_yield"] = float(yield_model.predict(X_yield)[0])
                except:
                    results["expected_yield"] = 0.0

        print(json.dumps(results))

    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    predict()
