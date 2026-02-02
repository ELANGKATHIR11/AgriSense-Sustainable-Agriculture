import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "enhanced_yield_prediction_model.pkl"
SCALER_PATH = MODEL_DIR / "enhanced_yield_prediction_scaler.pkl"
ENCODER_PATH = MODEL_DIR / "enhanced_yield_prediction_encoder.pkl"
TEXTURE_ENC_PATH = MODEL_DIR / "yield_soil_texture_encoder.pkl"


def predict():
    try:
        # Read Input
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"error": "No input data"}))
            sys.exit(1)

        data = json.loads(input_data)

        # Features expected by the NEW model:
        # ['Area_Ha', 'label_encoded', 'N', 'P', 'K', 'temperature', 'humidity', 'rainfall']

        # Required inputs from user
        # 'area' (in hectares or acres? let's assume m2 or Ha needed. Dataset has area_m2.
        # API likely sends 'area' and maybe 'unit'. Let's standardise to Ha.)
        # 'crop' (Crop name)

        required_inputs = [
            "crop",
            "area",
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "rainfall",
        ]

        # Check missing
        missing = [f for f in required_inputs if f not in data]
        if missing:
            # Try to fill missing env data with defaults if valid
            pass  # For now, strict check or default?
            # Let's enforce strictly as per request for "increased accuracy"
            print(json.dumps({"error": f"Missing fields: {missing}"}))
            sys.exit(1)

        # Prepare Inputs
        area = float(data["area"])
        # Handle unit if provided, else assume Ha or m2?
        # Dataset used m2 -> Ha.
        # Common user input is likely Acres or Hectares.
        # Let's assume input is Hectares for simplicity or check 'unit'.
        if data.get("area_unit", "").lower() == "acre":
            area = area * 0.404686
        elif data.get("area_unit", "").lower() == "m2":
            area = area / 10000.0

        crop = data["crop"]

        # Load artifacts
        if (
            not MODEL_PATH.exists()
            or not SCALER_PATH.exists()
            or not ENCODER_PATH.exists()
        ):
            print(
                json.dumps(
                    {"error": "Model files (model, scaler, or encoder) not found."}
                )
            )
            sys.exit(1)

        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)

        # Encode Crop
        # Handle unseen labels
        if crop not in encoder.classes_:
            # Fallback or error?
            # Try to map to closest or 'Rice' default?
            # Error is safer.
            print(
                json.dumps(
                    {
                        "error": f"Unknown crop: {crop}. Supported crops: {list(encoder.classes_[:5])}..."
                    }
                )
            )
            sys.exit(1)

        crop_encoded = encoder.transform([crop])[0]

        # Create Feature Vector
        # Order: ['Area_Ha', 'label_encoded', 'N', 'P', 'K', 'temperature', 'humidity', 'rainfall']
        features = pd.DataFrame(
            [
                [
                    area,
                    crop_encoded,
                    float(data["N"]),
                    float(data["P"]),
                    float(data["K"]),
                    float(data["temperature"]),
                    float(data["humidity"]),
                    float(data["rainfall"]),
                ]
            ],
            columns=[
                "Area_Ha",
                "label_encoded",
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "rainfall",
                "rainfall",
            ],
        )

        # Handle Soil Texture
        if TEXTURE_ENC_PATH.exists():
            tex_enc = joblib.load(TEXTURE_ENC_PATH)
            # Default to "Loam"
            texture = data.get("soil_texture", "Loam")
            if texture not in tex_enc.classes_:
                texture = "Loam"
            features["soil_texture_encoded"] = tex_enc.transform([texture])[0]

            # Reorder if necessary? No, just append is usually fine if scaler expects it at end.
            # Scaler was fit on [base... texture_enc]. So DataFrame needs to be in that order.
            # features order is [base...]. Append creates new column at end. Correct.

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]

        # Prediction is in hg/ha (from training script)
        # Convert to tonnes/ha or kg/acre as fit.
        # hg/ha -> kg/ha: / 10
        # hg/ha -> tonnes/ha: / 10000

        yield_kg_ha = prediction / 10.0
        yield_ton_ha = prediction / 10000.0

        # Result
        result = {
            "predicted_yield": float(yield_ton_ha),
            "unit": "tonnes per hectare",
            "yield_kg_ha": float(yield_kg_ha),
            "status": "success",
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "status": "error"}))
        sys.exit(1)


if __name__ == "__main__":
    predict()
