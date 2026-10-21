import os
import json
import joblib
import pandas as pd
from app.core.config import settings

class YieldPredictor:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.cat_cols = []
        self.num_cols = []
        self.meta = {}
        self.load_model()

    def load_model(self):
        model_path = os.path.join(settings.MODELS_DIR, "artifacts", "crop_yield_model.joblib")
        meta_path = os.path.join(settings.MODELS_DIR, "metadata", "crop_yield_model.json")
        
        if os.path.exists(model_path):
            try:
                artifact = joblib.load(model_path)
                self.model = artifact["model"]
                self.encoder = artifact["encoder"]
                self.cat_cols = artifact["cat_cols"]
                self.num_cols = artifact["num_cols"]
                print(f"Crop Yield Regressor loaded from {model_path}!")
            except Exception as e:
                print(f"Error loading yield model: {e}")

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    self.meta = json.load(f)
            except Exception as e:
                print(f"Error loading yield meta: {e}")

    def predict(self, region: str, soil_type: str, crop: str, rainfall: float, temp: float, 
                fertilizer_used: bool, irrigation_used: bool, weather: str, days: int) -> float:
        if self.model is None or self.encoder is None:
            # Fallback baseline calculation
            return round(4.5 + (0.002 * rainfall) + (0.5 if fertilizer_used else 0.0) + (0.8 if irrigation_used else 0.0), 2)

        raw_df = pd.DataFrame([{
            "Region": region,
            "Soil_Type": soil_type,
            "Crop": crop,
            "Weather_Condition": weather,
            "Rainfall_mm": rainfall,
            "Temperature_Celsius": temp,
            "Fertilizer_Used": int(fertilizer_used),
            "Irrigation_Used": int(irrigation_used),
            "Days_to_Harvest": days
        }])

        encoded_cat = self.encoder.transform(raw_df[self.cat_cols])
        encoded_df = pd.DataFrame(encoded_cat, columns=self.cat_cols)
        X = pd.concat([encoded_df, raw_df[self.num_cols]], axis=1)

        pred = self.model.predict(X)[0]
        return round(float(pred), 2)

yield_predictor = YieldPredictor()
