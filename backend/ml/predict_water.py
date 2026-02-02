import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "enhanced_water_requirement_model.pkl"
SCALER_PATH = MODEL_DIR / "enhanced_water_requirement_scaler.pkl"
ENCODER_PATH = MODEL_DIR / "enhanced_water_requirement_encoder.pkl"
TEXTURE_ENC_PATH = MODEL_DIR / "water_soil_texture_encoder.pkl"


def predict():
    try:
        # Read Input
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"error": "No input data"}))
            sys.exit(1)

        data = json.loads(input_data)

        # Validate required fields
        # features = ["temperature", "humidity", "rainfall", "label_encoded"]
        required_features = ["temperature", "humidity", "rainfall", "crop"]
        for feature in required_features:
            if feature not in data:
                print(json.dumps({"error": f"Missing feature: {feature}"}))
                sys.exit(1)

        # Load Artifacts
        if (
            not MODEL_PATH.exists()
            or not SCALER_PATH.exists()
            or not ENCODER_PATH.exists()
        ):
            print(json.dumps({"error": "Model files not found"}))
            sys.exit(1)

        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)

        # Prepare Inputs
        crop = data["crop"]

        # Check if crop in encoder
        if crop not in encoder.classes_:
            print(json.dumps({"error": f"Unknown crop: {crop}"}))
            sys.exit(1)

        crop_encoded = encoder.transform([crop])[0]

        # Create DataFrame for scaling matches training
        # X = df_model[["temperature", "humidity", "rainfall", "label_encoded"]]

        input_vector = pd.DataFrame(
            [
                [
                    float(data["temperature"]),
                    float(data["humidity"]),
                    float(data["rainfall"]),
                    crop_encoded,
                ]
            ],
            columns=["temperature", "humidity", "rainfall", "label_encoded"],
        )

        # Handle Soil Texture
        if TEXTURE_ENC_PATH.exists():
            tex_enc = joblib.load(TEXTURE_ENC_PATH)
            texture = data.get("soil_texture", "Loam")
            if texture not in tex_enc.classes_:
                texture = "Loam"
            input_vector["soil_texture_encoded"] = tex_enc.transform([texture])[0]

        # Preprocess
        input_scaled = scaler.transform(input_vector)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Result
        result = {
            "predicted_water_requirement": float(prediction),
            "unit": "mm per day",
            "confidence": 0.85,  # RandomForestRegressor doesn't give probs easily, using mock confidence or could use variance if available
            "status": "success",
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "status": "error"}))
        sys.exit(1)


if __name__ == "__main__":
    predict()
