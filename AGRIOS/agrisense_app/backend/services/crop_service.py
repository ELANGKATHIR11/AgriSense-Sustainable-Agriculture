import logging
import os
from typing import Any, Dict, List

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class CropService:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "models", "crop_recommendation_model.pkl")
        self.model_data = None
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at {self.model_path}")
                return

            self.model_data = joblib.load(self.model_path)
            # The structure of the pkl file from 'agrisense claude'
            self.model = self.model_data["models"][self.model_data["best_model_name"]]["model"]
            self.scaler = self.model_data["scaler"]
            self.label_encoder = self.model_data["label_encoder"]
            self.feature_names = self.model_data["feature_names"]
            logger.info(f"Crop Recommendation Model loaded successfully: {self.model_data.get('best_model_name')}")
        except Exception as e:
            logger.error(f"Failed to load crop recommendation model: {e}")

    def predict_crop(self, soil_data: Dict[str, float], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Predict suitable crops based on soil parameters.

        Args:
            soil_data: Dictionary containing:
                pH, N, P, K, Fe, Mn, Zn, Cu, B, Water, Moisture, Temperature, Rainfall
            top_n: Number of recommendations to return

        Returns:
            List of dictionaries with 'rank', 'crop', and 'suitability' keys.
        """
        if self.model is None:
            raise ValueError("Model not loaded properly")

        # Validate input keys match feature names if possible, or mapping
        # content of feature_names from analysis: ['pH', 'N', 'P', 'K', 'Fe', 'Mn', 'Zn', 'Cu', 'B', 'Water', 'Moisture', 'Temperature', 'Rainfall']
        # Ensure input dictionary has these keys

        try:
            input_df = pd.DataFrame([soil_data])

            # Ensure columns are in the correct order for the scaler
            # (Assuming input_df has correct keys, but let's be safe if feature_names is available)
            if self.feature_names:
                input_df = input_df[self.feature_names]

            input_scaled = self.scaler.transform(input_df)

            # Get predictions
            probabilities = self.model.predict_proba(input_scaled)[0]

            # Get top N indices
            top_indices = probabilities.argsort()[-top_n:][::-1]

            results = []
            for rank, idx in enumerate(top_indices, 1):
                crop_name = self.label_encoder.inverse_transform([idx])[0]
                suitability = round(probabilities[idx] * 100, 2)
                results.append({"rank": rank, "crop": crop_name, "suitability": suitability})

            return results

        except Exception as e:
            logger.error(f"Error during crop prediction: {e}")
            raise e


crop_service = CropService()
