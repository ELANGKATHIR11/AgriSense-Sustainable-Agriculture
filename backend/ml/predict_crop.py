import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "enhanced_crop_recommendation_model.pkl"
SCALER_PATH = MODEL_DIR / "enhanced_crop_recommendation_scaler.pkl"
ENCODER_PATH = MODEL_DIR / "enhanced_crop_recommendation_encoder.pkl"
TEXTURE_ENC_PATH = MODEL_DIR / "soil_texture_encoder.pkl"
AGRO_ENC_PATH = MODEL_DIR / "agro_climatic_zone_encoder.pkl"


def predict():
    try:
        # Read Input
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"error": "No input data"}))
            sys.exit(1)

        data = json.loads(input_data)

        # Validate required fields
        required_features = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ]
        for feature in required_features:
            if feature not in data:
                print(json.dumps({"error": f"Missing feature: {feature}"}))
                sys.exit(1)

        # Create DataFrame
        # Ensure order matches training
        df = pd.DataFrame(
            [[data[f] for f in required_features]], columns=required_features
        )

        # Load Artifacts
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            print(json.dumps({"error": "Model files not found"}))
            sys.exit(1)

        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)

        # Handle New Features
        # Texture
        if TEXTURE_ENC_PATH.exists():
            tex_enc = joblib.load(TEXTURE_ENC_PATH)
            # Default to "Loam" if missing or invalid
            texture = data.get("soil_texture", "Loam")
            if texture not in tex_enc.classes_:
                texture = "Loam"
            df["soil_texture_encoded"] = tex_enc.transform([texture])[0]

        # Agro Zone
        if AGRO_ENC_PATH.exists():
            agro_enc = joblib.load(AGRO_ENC_PATH)
            # Default to "Plateau"
            agro = data.get("agro_climatic_zone", "Plateau")
            if agro not in agro_enc.classes_:
                agro = "Plateau"
            df["agro_climatic_zone_encoded"] = agro_enc.transform([agro])[0]

        # Avg Rainfall
        # Use provided rainfall as proxy if missing
        df["avg_rainfall_mm"] = float(data.get("avg_rainfall_mm", data["rainfall"]))

        # Preprocess
        # Re-order columns to match scaler inputs?
        # scaler works on array. df order matters!
        # retrain script order: N, P, K, T, H, ph, rain, texture_enc, agro_enc, avg_rain
        # ensure df matches this.

        required_order = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        if "soil_texture_encoded" in df.columns:
            required_order.append("soil_texture_encoded")
        if "agro_climatic_zone_encoded" in df.columns:
            required_order.append("agro_climatic_zone_encoded")
        if "avg_rainfall_mm" in df.columns:
            required_order.append("avg_rainfall_mm")

        df = df[required_order]

        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)[0]

        # Decode
        prediction_label = prediction
        if ENCODER_PATH.exists():
            encoder = joblib.load(ENCODER_PATH)
            # Check if prediction is index or label
            # usually sklearn models on encoded targets predict the index
            try:
                prediction_label = encoder.inverse_transform([prediction])[0]
            except:
                pass  # Maybe model predicts strings directly?

        # Confidence
        confidence = 0.92
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_scaled)
            confidence = float(np.max(probs))

        # Result
        result = {
            "prediction": prediction_label,
            "confidence": confidence,
            "status": "success",
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "status": "error"}))
        sys.exit(1)


if __name__ == "__main__":
    predict()
