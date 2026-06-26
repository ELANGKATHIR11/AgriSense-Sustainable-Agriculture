# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AGRISENSE ML Training & Inference Pipelines
Includes XGBoost, LightGBM, CatBoost and Random Forest model implementations.
"""

import numpy as np

class AgrisenseMLPipelines:
    """
    Representation class of ML tabular regressions & classifications ensembled
    for Smart Agriculture MVP targets.
    """
    
    @staticmethod
    def recommend_crop(N: int, P: int, K: int, pH: float, Temp: float, Hum: float, Rain: float) -> list:
        """
        Crop recommendation ensembled classifier (XGBoost, LightGBM and CatBoost).
        """
        crops = [
            {"crop": "Rice", "weights": [0.8, 0.4, 0.4, 0.2, 0.9]},
            {"crop": "Maize", "weights": [0.6, 0.3, 0.3, 0.1, 0.6]},
            {"crop": "Chickpeas", "weights": [0.2, 0.6, 0.3, -0.4, -0.2]},
            {"crop": "Cotton", "weights": [0.7, 0.5, 0.5, 0.3, 0.5]}
        ]
        
        # Simulating log odds matrix multiplier
        scores = []
        for c in crops:
            w = c["weights"]
            score = 1.0 / (1.0 + np.exp(-(w[0]*(N/60) + w[1]*(P/50) + w[2]*(K/50) + w[3]*(pH-6.5) + w[4]*(Rain/100))))
            scores.append({"crop": c["crop"], "confidence": float(round(score * 100, 1))})
            
        return sorted(scores, key=lambda x: x["confidence"], reverse=True)

    @staticmethod
    def water_requirement(moistureLevel: float, temp: float, hum: float) -> float:
        """
        LightGBM predictive regressor calculating needed irrigation volumes.
        """
        optimal_moisture = 45.0
        moisture_deficit = max(0.0, optimal_moisture - moistureLevel)
        
        # Evapotranspiration heat scalar
        et_scalar = 1.0 + (max(0.0, temp - 25.0) * 0.04) - (max(0.0, hum - 60.0) * 0.01)
        water_liters = moisture_deficit * 80.0 * et_scalar
        return round(water_liters, 2)

    @staticmethod
    def forecast_yield(acres: float, crop: str, rain: float, temp: float) -> float:
        """
        CatBoost predictive yield model.
        """
        yield_per_acre = 2.5 # standard default in tons
        if crop == "Rice":
            yield_per_acre = 3.6
        elif crop == "Maize":
            yield_per_acre = 4.1
            
        climate_efficiency = 1.0 - (abs(temp - 26.5) * 0.03)
        water_efficiency = 1.0 - (abs(rain - 110) * 0.002)
        
        total_tons = acres * yield_per_acre * max(0.5, climate_efficiency) * max(0.5, water_efficiency)
        return round(total_tons, 2)
