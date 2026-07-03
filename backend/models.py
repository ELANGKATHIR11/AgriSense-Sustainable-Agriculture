# -*- coding: utf-8 -*-
"""
Models bridge file — re-exports all ORM models from backend/database/models
for backwards-compatibility with existing import paths.
"""

from backend.database.models import (  # noqa: F401  (re-exports)
    Role,
    Permission,
    User,
    Farm,
    Field,
    FarmBoundary,
    Sensor,
    SensorReading,
    Device,
    Weather,
    SatelliteMetadata,
    SatelliteTile,
    DroneImage,
    CropHealth,
    DiseaseDetection,
    WeedDetection,
    YieldPrediction,
    Recommendation,
    Notification,
    Task,
    AIAgent,
    Chat,
    Document,
    AuditLog,
    TwinState,
    ModelRegistry,
    PredictionLog,
    MarketProduct,
    Vendor,
    Subscription,
    LicenseKey,
)
