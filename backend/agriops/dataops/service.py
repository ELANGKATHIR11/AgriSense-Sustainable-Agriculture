# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps DataOps Service Layer
Handles dataset validation, data profiling, lineage tracking, and data drift analysis.
"""

import logging
from typing import Dict, Any
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.database.models import SensorReading, SatelliteMetadata, DroneImage
from backend.agriops.common.event_bus import event_bus
from backend.agriops.telemetry.tracer import trace_span
from backend.mlops.drift_detector import DriftDetector

logger = logging.getLogger("AgriOps.DataOps")


class DataOpsService:
    def __init__(self):
        self.drift_detector = DriftDetector()

    @trace_span("DataOps.ValidateSensorTelemetry")
    async def validate_and_ingest_metrics(
        self, db: Session, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates inbound telemetry payload and checks against historical reference distributions to detect data drift.
        """
        # Calculate quality score (0.0 to 1.0)
        missing_fields = 0
        total_fields = 7
        req_fields = [
            "soil_moisture",
            "temperature",
            "humidity",
            "ph",
            "nitrogen",
            "phosphorus",
            "potassium",
        ]

        for field in req_fields:
            if payload.get(field) is None:
                missing_fields += 1

        quality_score = 1.0 - (missing_fields / total_fields)

        # Pull last 100 historical readings for drift calculation
        history = (
            db.query(SensorReading)
            .order_by(SensorReading.timestamp.desc())
            .limit(100)
            .all()
        )
        drift_report = {
            "drift_detected": False,
            "msg": "Insufficient history for drift analysis",
        }

        if len(history) >= 10:
            # Prepare arrays for drift analysis
            ref_dict = {
                "soil_moisture": [
                    h.soil_moisture for h in history if h.soil_moisture is not None
                ],
                "temperature": [
                    h.temperature for h in history if h.temperature is not None
                ],
                "humidity": [h.humidity for h in history if h.humidity is not None],
            }

            curr_dict = {
                "soil_moisture": [payload.get("soil_moisture", 0.0)],
                "temperature": [payload.get("temperature", 0.0)],
                "humidity": [payload.get("humidity", 0.0)],
            }

            try:
                drift_res = self.drift_detector.analyze_dataset_drift(
                    ref_dict, curr_dict
                )
                drift_report = {
                    "drift_detected": drift_res["drift_detected"],
                    "drifted_features": drift_res["drifted_features_count"],
                    "metrics": drift_res["metrics"],
                }
            except Exception as e:
                logger.error(f"Error computing telemetry drift: {e}")

        validation_result = {
            "valid": quality_score > 0.7,
            "quality_score": round(quality_score, 2),
            "drift_analysis": drift_report,
            "validated_at": datetime.now(timezone.utc).isoformat() + "Z",
        }

        # Fire event
        await event_bus.publish(
            "DatasetValidated",
            {
                "source": "sensor_telemetry",
                "quality_score": quality_score,
                "drift_detected": drift_report.get("drift_detected", False),
            },
        )

        return validation_result

    @trace_span("DataOps.GetDatasetRegistry")
    def get_dataset_registry_summary(self, db: Session) -> Dict[str, Any]:
        """
        Compiles metadata regarding active telemetry records, drone surveys, and satellite scans.
        """
        sensor_readings_count = db.query(SensorReading).count()
        satellite_scans_count = db.query(SatelliteMetadata).count()
        drone_images_count = db.query(DroneImage).count()

        # Retrieve last drone image path for lineage demonstration
        latest_drone = (
            db.query(DroneImage).order_by(DroneImage.flight_date.desc()).first()
        )
        latest_satellite = (
            db.query(SatelliteMetadata)
            .order_by(SatelliteMetadata.acquisition_date.desc())
            .first()
        )

        lineage = []
        if latest_drone:
            lineage.append(
                {
                    "type": "Drone",
                    "source": latest_drone.file_path,
                    "timestamp": latest_drone.flight_date.isoformat() + "Z"
                    if latest_drone.flight_date
                    else None,
                    "description": f"Drone coverage area assessment. Resolution: {latest_drone.resolution_cm}cm",
                }
            )
        if latest_satellite:
            lineage.append(
                {
                    "type": "Satellite",
                    "source": latest_satellite.provider,
                    "timestamp": latest_satellite.acquisition_date.isoformat() + "Z"
                    if latest_satellite.acquisition_date
                    else None,
                    "description": f"Satellite data registry scan. Cloud cover: {latest_satellite.cloud_cover}%",
                }
            )

        return {
            "datasets": [
                {
                    "name": "IoT-Sensor-Telemetry",
                    "type": "Tabular",
                    "count": sensor_readings_count,
                    "format": "Real-time MQTT",
                },
                {
                    "name": "Sentinel-Satellite-Imagery",
                    "type": "Raster",
                    "count": satellite_scans_count,
                    "format": "TIFF",
                },
                {
                    "name": "Drone-Survey-Orthomosaics",
                    "type": "Image",
                    "count": drone_images_count,
                    "format": "JPG",
                },
            ],
            "lineages": lineage,
            "registry_version": "5.0.0",
        }


dataops_service = DataOpsService()
