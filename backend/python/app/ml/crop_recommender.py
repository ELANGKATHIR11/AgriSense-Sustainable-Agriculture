import os
import json
import joblib
import numpy as np
from app.core.config import settings

# Comprehensive Indian agronomic crop knowledge profiles
CROP_PROFILES = {
    "Rice": {
        "scientificName": "Oryza sativa",
        "opt_N": 120, "opt_P": 60, "opt_K": 60, "opt_temp": 25, "opt_hum": 80, "opt_ph": 6.5, "opt_rain": 1200,
        "waterRequirementMm": 1250, "growthDurationDays": 135, "expectedYieldTonHa": 5.8, "dripSavingsPct": 38.0
    },
    "Wheat": {
        "scientificName": "Triticum aestivum",
        "opt_N": 120, "opt_P": 60, "opt_K": 40, "opt_temp": 20, "opt_hum": 60, "opt_ph": 6.8, "opt_rain": 600,
        "waterRequirementMm": 550, "growthDurationDays": 120, "expectedYieldTonHa": 4.5, "dripSavingsPct": 42.0
    },
    "Maize": {
        "scientificName": "Zea mays",
        "opt_N": 120, "opt_P": 60, "opt_K": 50, "opt_temp": 26, "opt_hum": 65, "opt_ph": 6.5, "opt_rain": 750,
        "waterRequirementMm": 600, "growthDurationDays": 105, "expectedYieldTonHa": 6.2, "dripSavingsPct": 45.0
    },
    "Cotton": {
        "scientificName": "Gossypium hirsutum",
        "opt_N": 100, "opt_P": 50, "opt_K": 50, "opt_temp": 28, "opt_hum": 55, "opt_ph": 7.0, "opt_rain": 700,
        "waterRequirementMm": 700, "growthDurationDays": 160, "expectedYieldTonHa": 2.8, "dripSavingsPct": 52.0
    },
    "Sugarcane": {
        "scientificName": "Saccharum officinarum",
        "opt_N": 250, "opt_P": 100, "opt_K": 120, "opt_temp": 30, "opt_hum": 75, "opt_ph": 6.8, "opt_rain": 1500,
        "waterRequirementMm": 1800, "growthDurationDays": 330, "expectedYieldTonHa": 85.0, "dripSavingsPct": 55.0
    },
    "Chickpea": {
        "scientificName": "Cicer arietinum",
        "opt_N": 30, "opt_P": 60, "opt_K": 30, "opt_temp": 22, "opt_hum": 50, "opt_ph": 7.2, "opt_rain": 400,
        "waterRequirementMm": 350, "growthDurationDays": 95, "expectedYieldTonHa": 1.9, "dripSavingsPct": 40.0
    },
    "Groundnut": {
        "scientificName": "Arachis hypogaea",
        "opt_N": 25, "opt_P": 50, "opt_K": 40, "opt_temp": 27, "opt_hum": 65, "opt_ph": 6.4, "opt_rain": 550,
        "waterRequirementMm": 500, "growthDurationDays": 115, "expectedYieldTonHa": 2.5, "dripSavingsPct": 48.0
    },
    "Banana": {
        "scientificName": "Musa acuminata",
        "opt_N": 200, "opt_P": 70, "opt_K": 300, "opt_temp": 28, "opt_hum": 82, "opt_ph": 6.5, "opt_rain": 1400,
        "waterRequirementMm": 1600, "growthDurationDays": 300, "expectedYieldTonHa": 42.0, "dripSavingsPct": 50.0
    },
    "Tomato": {
        "scientificName": "Solanum lycopersicum",
        "opt_N": 150, "opt_P": 80, "opt_K": 100, "opt_temp": 24, "opt_hum": 70, "opt_ph": 6.3, "opt_rain": 650,
        "waterRequirementMm": 600, "growthDurationDays": 90, "expectedYieldTonHa": 35.0, "dripSavingsPct": 60.0
    },
    "Potato": {
        "scientificName": "Solanum tuberosum",
        "opt_N": 160, "opt_P": 100, "opt_K": 120, "opt_temp": 19, "opt_hum": 65, "opt_ph": 5.8, "opt_rain": 500,
        "waterRequirementMm": 500, "growthDurationDays": 100, "expectedYieldTonHa": 25.0, "dripSavingsPct": 45.0
    }
}

class CropRecommender:
    def __init__(self):
        self.model = None
        self.meta = {}
        self.load_model()

    def load_model(self):
        model_path = os.path.join(settings.MODELS_DIR, "artifacts", "crop_recommendation_rf.joblib")
        meta_path = os.path.join(settings.MODELS_DIR, "metadata", "crop_recommendation_rf.json")
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading crop recommendation model: {e}")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    self.meta = json.load(f)
            except Exception as e:
                print(f"Error loading crop meta: {e}")

    def recommend(self, n: float, p: float, k: float, temp: float, hum: float, ph: float, rain: float) -> list:
        scores = []
        for crop_name, prof in CROP_PROFILES.items():
            # Normalized Euclidean distance across 7 agronomic dimensions
            d_n = ((n - prof["opt_N"]) / 150.0) ** 2
            d_p = ((p - prof["opt_P"]) / 80.0) ** 2
            d_k = ((k - prof["opt_K"]) / 120.0) ** 2
            d_t = ((temp - prof["opt_temp"]) / 15.0) ** 2
            d_h = ((hum - prof["opt_hum"]) / 30.0) ** 2
            d_ph = ((ph - prof["opt_ph"]) / 1.5) ** 2
            d_r = ((rain - prof["opt_rain"]) / 600.0) ** 2
            
            dist = np.sqrt(d_n + d_p + d_k + d_t + d_h + d_ph + d_r)
            # Convert distance to calibrated similarity confidence
            conf = float(np.exp(-dist / 2.0))
            
            # Find limiting factor
            factors = {
                "Nitrogen deficit": abs(n - prof["opt_N"]) / 150.0,
                "Phosphorus deficit": abs(p - prof["opt_P"]) / 80.0,
                "Potassium deficit": abs(k - prof["opt_K"]) / 120.0,
                "Thermal stress": abs(temp - prof["opt_temp"]) / 15.0,
                "Rainfall divergence": abs(rain - prof["opt_rain"]) / 600.0,
                "Soil pH variance": abs(ph - prof["opt_ph"]) / 1.5
            }
            limiting = max(factors.items(), key=lambda x: x[1])[0]

            scores.append({
                "crop_name": crop_name,
                "profile": prof,
                "conf": conf,
                "limiting": limiting
            })

        scores.sort(key=lambda x: x["conf"], reverse=True)
        top_crops = scores[:3]

        total_top_conf = sum(x["conf"] for x in top_crops)
        recommendations = []
        for rank, item in enumerate(top_crops, 1):
            prof = item["profile"]
            calibrated_pct = round((item["conf"] / total_top_conf) * 100.0, 1) if total_top_conf > 0 else 33.3
            recommendations.append({
                "cropKey": item["crop_name"].lower().replace(" ", "_"),
                "cropName": item["crop_name"],
                "scientificName": prof["scientificName"],
                "rank": rank,
                "confidencePct": calibrated_pct,
                "waterRequirementMm": prof["waterRequirementMm"],
                "growthDurationDays": prof["growthDurationDays"],
                "expectedYieldTonHa": prof["expectedYieldTonHa"],
                "dripSavingsPct": prof["dripSavingsPct"],
                "limitingFactor": item["limiting"]
            })

        return recommendations

crop_recommender = CropRecommender()
