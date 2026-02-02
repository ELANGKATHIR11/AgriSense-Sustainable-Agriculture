import argparse
import json
import os
import sys

import joblib
import numpy as np


def predict(model_path, temp, humidity, rainfall, ph):
    if not os.path.exists(model_path):
        print(
            json.dumps(
                {
                    "error": "Model file not found. Please train the model first."
                }
            )
        )
        sys.exit(1)

    try:
        model = joblib.load(model_path)

        # Prepare input
        features = np.array([[temp, humidity, rainfall, ph]])

        # Predict class and probability
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Class 1 is typically "Disease Present" in this dataset
        risk_score = probability[1]  # Probability of class 1

        result = {
            "disease_present_prediction": int(prediction),
            "risk_score": float(risk_score),
            "risk_level": (
                "High"
                if risk_score > 0.7
                else ("Medium" if risk_score > 0.3 else "Low")
            ),
            "input": {
                "temperature": temp,
                "humidity": humidity,
                "rainfall": rainfall,
                "soil_pH": ph,
            },
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Environmental Disease Risk"
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(
        base_dir, "models", "environmental_model.joblib"
    )

    parser.add_argument(
        "--model", type=str, default=default_model, help="Path to model file"
    )
    parser.add_argument(
        "--temp", type=float, required=True, help="Temperature (C)"
    )
    parser.add_argument(
        "--humidity", type=float, required=True, help="Humidity (%)"
    )
    parser.add_argument(
        "--rainfall", type=float, required=True, help="Rainfall (mm)"
    )
    parser.add_argument("--ph", type=float, required=True, help="Soil pH")

    args = parser.parse_args()

    predict(args.model, args.temp, args.humidity, args.rainfall, args.ph)
