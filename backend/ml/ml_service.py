#!/usr/bin/env python3
"""
Python ML Service Bridge for Node.js Backend
Provides REST API endpoints for ML model inference
"""

import os
import sys
from pathlib import Path

import joblib  # type: ignore
import numpy as np
from flask import Flask, jsonify, request  # type: ignore
from flask_cors import CORS  # type: ignore

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


app = Flask(__name__)
CORS(app)

# Model paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"


class MLModelService:
    """Service for loading and using ML models"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.load_all_models()

    def load_all_models(self):
        """Load all available ML models"""
        print("Loading ML models...")

        # Crop Recommendation Model
        crop_model_path = MODELS_DIR / (
            "enhanced_crop_recommendation_model.pkl"
        )
        crop_scaler_path = MODELS_DIR / (
            "enhanced_crop_recommendation_scaler.pkl"
        )
        crop_encoder_path = MODELS_DIR / (
            "enhanced_crop_recommendation_encoder.pkl"
        )

        if crop_model_path.exists():
            try:
                self.models["crop_recommendation"] = joblib.load(
                    crop_model_path
                )
                self.scalers["crop_recommendation"] = joblib.load(
                    crop_scaler_path
                )
                self.encoders["crop_recommendation"] = joblib.load(
                    crop_encoder_path
                )
                print("✅ Loaded crop recommendation model")
            except Exception as e:
                print(f"⚠️  Failed to load crop recommendation model: {e}")

        # Yield Prediction Model
        yield_model_path = MODELS_DIR / "enhanced_yield_prediction_model.pkl"
        yield_scaler_path = MODELS_DIR / (
            "enhanced_yield_prediction_scaler.pkl"
        )

        if yield_model_path.exists():
            try:
                self.models["yield_prediction"] = joblib.load(yield_model_path)
                self.scalers["yield_prediction"] = joblib.load(
                    yield_scaler_path
                )
                print("✅ Loaded yield prediction model")
            except Exception as e:
                print(f"⚠️  Failed to load yield prediction model: {e}")

        # Crop Type Classification Model
        crop_type_model_path = MODELS_DIR / (
            "enhanced_crop_type_classification_model.pkl"
        )
        crop_type_scaler_path = MODELS_DIR / (
            "enhanced_crop_type_classification_scaler.pkl"
        )
        crop_type_encoder_path = MODELS_DIR / (
            "enhanced_crop_type_classification_encoder.pkl"
        )

        if crop_type_model_path.exists():
            try:
                self.models["crop_type"] = joblib.load(crop_type_model_path)
                self.scalers["crop_type"] = joblib.load(crop_type_scaler_path)
                self.encoders["crop_type"] = joblib.load(
                    crop_type_encoder_path
                )
                print("✅ Loaded crop type classification model")
            except Exception as e:
                print(f"⚠️  Failed to load crop type model: {e}")

        # Water Requirement Model
        water_model_path = MODELS_DIR / "enhanced_water_requirement_model.pkl"
        water_scaler_path = MODELS_DIR / (
            "enhanced_water_requirement_scaler.pkl"
        )

        if water_model_path.exists():
            try:
                self.models["water_requirement"] = joblib.load(
                    water_model_path
                )
                self.scalers["water_requirement"] = joblib.load(
                    water_scaler_path
                )
                print("✅ Loaded water requirement model")
            except Exception as e:
                print(f"⚠️  Failed to load water requirement model: {e}")

        # Season Classification Model
        season_model_path = MODELS_DIR / (
            "enhanced_season_classification_model.pkl"
        )
        season_scaler_path = MODELS_DIR / (
            "enhanced_season_classification_scaler.pkl"
        )
        season_encoder_path = MODELS_DIR / (
            "enhanced_season_classification_encoder.pkl"
        )

        if season_model_path.exists():
            try:
                self.models["season_classification"] = joblib.load(
                    season_model_path
                )
                self.scalers["season_classification"] = joblib.load(
                    season_scaler_path
                )
                self.encoders["season_classification"] = joblib.load(
                    season_encoder_path
                )
                print("✅ Loaded season classification model")
            except Exception as e:
                print(f"⚠️  Failed to load season classification model: {e}")

        print(f"Loaded {len(self.models)} models")

    def predict_crop_recommendation(self, data):
        """Predict crop recommendation"""
        if "crop_recommendation" not in self.models:
            raise ValueError("Crop recommendation model not loaded")

        # Extract features
        features = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ]
        X = np.array([[data.get(f, 0) for f in features]])

        # Scale features
        X_scaled = self.scalers["crop_recommendation"].transform(X)

        # Predict
        prediction = self.models["crop_recommendation"].predict(X_scaled)[0]
        probabilities = self.models["crop_recommendation"].predict_proba(
            X_scaled
        )[0]

        # Decode crop name
        crop_name = self.encoders["crop_recommendation"].inverse_transform(
            [prediction]
        )[0]
        confidence = float(max(probabilities))

        # Get top alternatives
        top_indices = np.argsort(probabilities)[-4:][::-1]
        alternatives = [
            self.encoders["crop_recommendation"].inverse_transform([idx])[0]
            for idx in top_indices
            if idx != prediction
        ][:3]

        return {
            "recommended_crop": crop_name,
            "confidence": confidence,
            "alternatives": alternatives,
        }

    def predict_yield(self, data):
        """Predict crop yield"""
        if "yield_prediction" not in self.models:
            raise ValueError("Yield prediction model not loaded")

        # Extract features
        features = [
            "N",
            "P",
            "K",
            "temperature",
            "rainfall",
            "water_requirement",
            "growth_duration",
        ]
        X = np.array([[data.get(f, 0) for f in features]])

        # Scale features
        X_scaled = self.scalers["yield_prediction"].transform(X)

        # Predict
        prediction = self.models["yield_prediction"].predict(X_scaled)[0]

        return {
            "predicted_yield": float(prediction),
            "unit": "tons per hectare",
        }

    def predict_crop_type(self, data):
        """Predict crop type classification"""
        if "crop_type" not in self.models:
            raise ValueError("Crop type model not loaded")

        # Extract features
        features = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ]
        X = np.array([[data.get(f, 0) for f in features]])

        # Scale features
        X_scaled = self.scalers["crop_type"].transform(X)

        # Predict
        prediction = self.models["crop_type"].predict(X_scaled)[0]
        probabilities = self.models["crop_type"].predict_proba(X_scaled)[0]

        # Decode crop type
        crop_type = self.encoders["crop_type"].inverse_transform([prediction])[
            0
        ]
        confidence = float(max(probabilities))

        return {"crop_type": crop_type, "confidence": confidence}

    def predict_water_requirement(self, data):
        """Predict water requirement"""
        if "water_requirement" not in self.models:
            raise ValueError("Water requirement model not loaded")

        # Extract features
        features = ["temperature", "humidity", "rainfall", "growth_duration"]
        X = np.array([[data.get(f, 0) for f in features]])

        # Scale features
        X_scaled = self.scalers["water_requirement"].transform(X)

        # Predict
        prediction = self.models["water_requirement"].predict(X_scaled)[0]

        return {"water_requirement": float(prediction), "unit": "mm per day"}

    def predict_season_classification(self, data):
        """Predict season classification"""
        if "season_classification" not in self.models:
            raise ValueError("Season classification model not loaded")

        # Extract features
        features = ["temperature", "rainfall", "humidity"]
        X = np.array([[data.get(f, 0) for f in features]])

        # Scale features
        X_scaled = self.scalers["season_classification"].transform(X)

        # Predict
        prediction = self.models["season_classification"].predict(X_scaled)[0]
        probabilities = self.models["season_classification"].predict_proba(
            X_scaled
        )[0]

        # Decode season
        season = self.encoders["season_classification"].inverse_transform(
            [prediction]
        )[0]
        confidence = float(max(probabilities))

        return {"season": season, "confidence": confidence}


ml_service = MLModelService()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "OK",
            "models_loaded": len(ml_service.models),
            "available_models": list(ml_service.models.keys()),
        }
    )


@app.route("/predict/crop-recommendation", methods=["POST"])
def predict_crop():
    """Crop recommendation endpoint"""
    try:
        data = request.json
        result = ml_service.predict_crop_recommendation(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/yield", methods=["POST"])
def predict_yield():
    """Yield prediction endpoint"""
    try:
        data = request.json
        result = ml_service.predict_yield(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/crop-type", methods=["POST"])
def predict_crop_type():
    """Crop type classification endpoint"""
    try:
        data = request.json
        result = ml_service.predict_crop_type(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/water-requirement", methods=["POST"])
def predict_water():
    """Water requirement prediction endpoint"""
    try:
        data = request.json
        result = ml_service.predict_water_requirement(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/season-classification", methods=["POST"])
def predict_season():
    """Season classification endpoint"""
    try:
        data = request.json
        result = ml_service.predict_season_classification(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("ML_SERVICE_PORT", 5001))
    print(f"🚀 Starting ML Service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
