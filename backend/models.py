# -*- coding: utf-8 -*-
"""
Models bridge file to maintain backwards compatibility with existing imports.
Redirects to the new structured backend/database/models submodule.
"""
from backend.database.base import Base
from backend.database.models import (
    User, SensorReading, TwinState, ModelRegistry, PredictionLog,
    Farm, Field, Device, MarketProduct, Vendor, Subscription,
    LicenseKey, AuditLog, Notification, Role, Permission,
    FarmBoundary, Sensor, Weather, SatelliteMetadata, SatelliteTile,
    DroneImage, CropHealth, DiseaseDetection, WeedDetection,
    YieldPrediction, Recommendation, Task, AIAgent, Chat, Document
)
