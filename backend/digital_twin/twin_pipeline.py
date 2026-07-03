# -*- coding: utf-8 -*-
"""
AGRISENSE Digital Twin Sequence Pipeline - Upgraded v4.0
Combines EIF anomaly validation, FAO-56 physics modeling, TabPFN corrections,
FT-Transformer yield prediction, and residual confidence estimation.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone

# Add root directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from ml.extended_isolation_forest import ExtendedIsolationForest
from backend.twin_engine import calculate_fao56_et0
from backend.ml.tabpfn_engine import load_or_init_tabpfn
from backend.ml.yield_transformer import load_or_train_yield

CLEANED_DATA_PATH = "AgriSense-Dataset/consolidated_agriculture_dataset.csv"


class DigitalTwinPipeline:
    def __init__(self):
        self.eif = ExtendedIsolationForest(n_estimators=50, max_samples=128)
        self.is_trained = False
        self.bootstrap_eif()

    def bootstrap_eif(self):
        """Train Extended Isolation Forest on healthy telemetry data."""
        try:
            if os.path.exists(CLEANED_DATA_PATH):
                df = pd.read_csv(CLEANED_DATA_PATH)
                df = df[df["source_file"] == "fertilizer_dataset.csv"].dropna(
                    how="all", axis=1
                )

                col_map = {
                    "nitrogen": "Nitrogen",
                    "phosphorous": "Phosphorous",
                    "potassium": "Potassium",
                    "temparature": "Temparature",
                    "humidity": "Humidity",
                    "ph": "pH",
                    "moisture": "Moisture",
                }
                for c_low, c_orig in col_map.items():
                    if c_orig not in df.columns and c_low in df.columns:
                        df[c_orig] = df[c_low]

                if "Moisture" not in df.columns:
                    df["Moisture"] = np.random.uniform(20, 60, len(df))

                features = df[
                    [
                        "Nitrogen",
                        "Phosphorous",
                        "Potassium",
                        "Temparature",
                        "Humidity",
                        "pH",
                        "Moisture",
                    ]
                ].values
                self.eif.fit(features)
                self.is_trained = True
                print("EIF anomaly model trained successfully on telemetry history.")
            else:
                raise FileNotFoundError()
        except Exception as e:
            print(
                f"Telemetry dataset load failed ({e}). Bootstrapping EIF with synthetic normal agricultural records..."
            )
            N = np.random.uniform(20, 140, 200)
            P = np.random.uniform(10, 90, 200)
            K = np.random.uniform(10, 90, 200)
            T = np.random.uniform(15, 40, 200)
            H = np.random.uniform(30, 90, 200)
            pH = np.random.uniform(5.5, 7.5, 200)
            M = np.random.uniform(20, 80, 200)

            features = np.column_stack([N, P, K, T, H, pH, M])
            self.eif.fit(features)
            self.is_trained = True

    def execute_pipeline(self, telemetry: dict) -> dict:
        """
        Executes sequence: Sensor -> EIF Validation -> Physics -> TabPFN/FT-Transformer Corrections
        """
        # 1. Prepare sensor vector
        vec = np.array(
            [
                [
                    telemetry.get("N", 50.0),
                    telemetry.get("P", 40.0),
                    telemetry.get("K", 40.0),
                    telemetry.get("temp", 28.0),
                    telemetry.get("humidity", 60.0),
                    telemetry.get("pH", 6.5),
                    telemetry.get("moisture", 38.0),
                ]
            ]
        )

        # 2. Run EIF Anomaly Validation
        anomaly_score = float(self.eif.compute_anomaly_score(vec)[0])
        has_anomaly = anomaly_score > 0.62
        anomaly_alerts = []
        if has_anomaly:
            anomaly_alerts.append(
                f"Extended Isolation Forest flagged sensor anomaly (score: {anomaly_score:.3f})."
            )

        # 3. Physics Layer (Penman-Monteith reference ET0)
        temp = telemetry.get("temp", 28.0)
        humidity = telemetry.get("humidity", 60.0)
        wind_speed = telemetry.get("wind_speed", 8.4)
        et0 = calculate_fao56_et0(temp, humidity, wind_speed)

        # 4. TabPFN Crop Correction Layer
        suitability_confidence = 0.90
        recommendation_notes = []

        try:
            tabpfn = load_or_init_tabpfn("crop_recommendation")
            crop_features = np.array(
                [
                    [
                        telemetry.get("N", 50.0),
                        telemetry.get("P", 40.0),
                        telemetry.get("K", 40.0),
                        telemetry.get("temp", 28.0),
                        telemetry.get("humidity", 60.0),
                        telemetry.get("pH", 6.5),
                        telemetry.get("rainfall", 100.0),
                    ]
                ]
            )
            probs = tabpfn.predict_proba(crop_features)[0]
            best_idx = np.argmax(probs)
            suitability_confidence = float(probs[best_idx])

            # Apply residual correction: decrease confidence if anomaly score is high
            correction = -0.18 * max(0.0, anomaly_score - 0.40)
            suitability_confidence = max(
                0.1, min(1.0, suitability_confidence + correction)
            )
            predicted_crop = str(tabpfn.classes_[best_idx])
            recommendation_notes.append(
                f"TabPFN correction recommends: {predicted_crop} (Conf: {suitability_confidence:.1%})"
            )
        except Exception as e:
            recommendation_notes.append(f"TabPFN correction bypassed: {str(e)}")

        # 5. FT-Transformer Yield Correction Layer
        try:
            yield_model = load_or_train_yield()
            x_yield = torch.tensor(
                [
                    [
                        1.0,
                        100.0,
                        temp,
                        telemetry.get("N", 50.0),
                        telemetry.get("P", 40.0),
                        telemetry.get("K", 40.0),
                    ]
                ],
                dtype=torch.float32,
            )
            with torch.no_grad():
                pred_val = float(yield_model(x_yield).cpu().numpy()[0])
            recommendation_notes.append(
                f"FT-Transformer projected yield index: {pred_val:.2f} tons/acre"
            )
        except Exception as e:
            recommendation_notes.append(
                f"FT-Transformer yield prediction bypassed: {str(e)}"
            )

        # 6. Uncertainty Estimation
        std_dev = math.sqrt(2.0 + (15.0 * anomaly_score))
        current_moisture = telemetry.get("moisture", 38.0)
        moisture_target = 42.0
        moisture_diff = max(0.0, moisture_target - current_moisture)
        water_deficit = round(moisture_diff * 90.0)

        lower_bound = max(0, round((moisture_diff - 1.96 * (std_dev / 10.0)) * 90.0))
        upper_bound = round((moisture_diff + 1.96 * (std_dev / 10.0)) * 90.0)

        # Build complete state
        twin_state = {
            "overallHealthScore": round(88 - (25 * anomaly_score)),
            "anomalyScore": round(anomaly_score, 3),
            "isAnomaly": has_anomaly,
            "alerts": anomaly_alerts,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "physicsModel": {
                "evapotranspirationET0": et0,
                "windSpeed": wind_speed,
                "waterDeficitLiters": water_deficit,
                "confidenceInterval": [lower_bound, upper_bound],
                "uncertaintyMarginLiters": round(1.96 * std_dev * 9.0),
            },
            "recommendationNotes": recommendation_notes,
        }

        return twin_state


twin_pipeline = DigitalTwinPipeline()
