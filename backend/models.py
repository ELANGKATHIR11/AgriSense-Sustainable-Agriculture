# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

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
