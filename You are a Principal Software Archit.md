You are a Principal Software Architect, Principal ML Engineer, MLOps Architect, Digital Twin Architect, FastAPI Expert, AI Systems Engineer, Agricultural Scientist, and Senior Full-Stack Engineer.

Your task is NOT to build a new project.

Your task is to AUDIT, REFACTOR, FIX, COMPLETE, and PRODUCTIONIZE the EXISTING AGRISENSE project.

====================================================
PRIMARY OBJECTIVE
====================================================

Analyze the entire existing codebase.

Preserve all working functionality.

Fix broken implementations.

Replace mock systems with real implementations.

Complete all missing backend services.

Complete ML pipelines.

Complete MLOps.

Complete Digital Twin backend.

Ensure seamless integration with the existing React frontend.

The final result must be ONE unified AGRISENSE platform.

====================================================
REFERENCE ARCHITECTURE
====================================================

Follow the AGRISENSE Blueprints.

Implement:

✓ AgriSense Core Platform
✓ AgriGPT
✓ LocalAgriBot
✓ Digital Twin
✓ MLOps Dashboard
✓ IoT Integration
✓ Analytics
✓ Reporting
✓ Training Pipelines

====================================================
PHASE 1
PROJECT AUDIT
====================================================

First perform a complete codebase audit.

Generate report:

1. Existing modules
2. Missing modules
3. Broken APIs
4. Mock implementations
5. Dead code
6. Duplicate code
7. Unused services
8. Missing integrations
9. Architecture violations
10. Security issues

Generate:

PROJECT_AUDIT_REPORT.md

====================================================
PHASE 2
FRONTEND FIXES
====================================================

Audit frontend.

Find:

- Direct fetch calls
- Hardcoded URLs
- API inconsistencies
- Duplicate state
- Component coupling
- Missing React Query hooks

Refactor frontend to:

UI
↓
React Query
↓
Service Layer
↓
API Client
↓
FastAPI Backend

Requirements:

No direct fetch calls in pages.

All requests must use:

src/services/

All endpoints must use:

src/api/endpoints.ts

All backend URLs must come from:

.env

Implement:

VITE_API_URL

USE_MOCK_DATA

Generate migration guide.

====================================================
PHASE 3
BACKEND RECONSTRUCTION
====================================================

Technology:

Python 3.12

FastAPI

SQLAlchemy

Pydantic V2

Alembic

SQLite

Structure:

backend/

api/
core/
database/
models/
schemas/
services/
repositories/
middleware/
utils/

====================================================
AUTHENTICATION
====================================================

Implement:

JWT Access

JWT Refresh

RBAC

Roles:

admin
farmer
researcher
student

====================================================
DATABASE
====================================================

Create and migrate:

users

sensor_readings

crop_recommendations

disease_predictions

weed_predictions

irrigation_predictions

yield_predictions

weather_cache

chat_history

twin_state

water_state

soil_state

crop_state

weather_state

disease_state

simulation_runs

alerts

model_registry

training_runs

model_metrics

audit_logs

====================================================
PHASE 4
ML ENGINE
====================================================

Build production ML engine.

====================================================
CROP RECOMMENDATION
====================================================

Models:

XGBoost
LightGBM
CatBoost

Inputs:

N
P
K
pH
Temperature
Humidity
Rainfall

Outputs:

Recommended Crops
Confidence
Explanation

Files:

crop_training.py
crop_inference.py

====================================================
IRRIGATION
====================================================

Models:

LightGBM
RandomForest

Outputs:

Water Requirement
Efficiency Score

Files:

irrigation_training.py
irrigation_inference.py

====================================================
YIELD PREDICTION
====================================================

Models:

XGBoost
RandomForest

Outputs:

Yield Forecast

Files:

yield_training.py
yield_inference.py

====================================================
ANOMALY DETECTION
====================================================

Model:

Isolation Forest

Detect:

Sensor Failure
Data Drift
Abnormal Conditions

====================================================
TRAINING PIPELINES
====================================================

Every model must support:

Train
Validate
Evaluate
Save
Version
Register
Deploy

Generate:

Training Metrics
Confusion Matrix
Feature Importance
Performance Reports

====================================================
PHASE 5
SMOLVLM INTEGRATION
====================================================

Replace all mock disease endpoints.

Current state:

Mock outputs.

Required state:

Image
↓
SmolVLM 3B
↓
Structured Analysis
↓
JSON Output

Implement:

POST /disease/analyze

POST /weed/analyze

Output:

Disease
Confidence
Severity
Symptoms
Recommendations

Use Ollama.

====================================================
PHASE 6
AGRIGPT
====================================================

Replace simple prompt-response endpoint.

Build:

AgriGPT

Using:

Qwen2.5-Coder 3B

Features:

Crop Advice
Disease Explanation
Yield Explanation
Irrigation Advice
Digital Twin Explanation
Report Generation

Implement:

Conversation Memory

Farm Context

User Context

====================================================
LOCAL RAG
====================================================

Use:

FAISS

Sentence Transformers

Knowledge Sources:

Crop Knowledge
Disease Knowledge
Irrigation Knowledge
Project Documentation

Pipeline:

User
↓
Retriever
↓
Context
↓
Qwen
↓
Answer

====================================================
PHASE 7
DIGITAL TWIN
====================================================

Build real Digital Twin backend.

Current frontend twin must become functional.

====================================================
WATER TWIN
====================================================

Implement:

FAO56 ET0

Penman Monteith

Water Balance

Moisture Forecast

Equation:

Moisture(t+1)

=
Moisture(t)
+ Rainfall
+ Irrigation
- ET0
- Drainage

====================================================
SOIL TWIN
====================================================

Implement:

NPK depletion

pH tracking

Fertility forecast

====================================================
CROP TWIN
====================================================

Implement:

Growth stage

Biomass

Health Score

Yield estimate

====================================================
WEATHER TWIN
====================================================

Implement:

Forecast ingestion

Weather risk

Rain prediction

====================================================
DISEASE TWIN
====================================================

Implement:

Disease probability

Disease spread risk

Outbreak prediction

====================================================
FARM TWIN
====================================================

Combine all twins.

Generate:

Farm Health Score

Risk Score

Yield Score

Sustainability Score

====================================================
SIMULATION ENGINE
====================================================

Build:

What-if simulations

Scenarios:

No irrigation

Heavy rain

Drought

Disease outbreak

Nutrient deficiency

Outputs:

Future twin state

Yield impact

Risk impact

====================================================
PHASE 8
MLOPS PLATFORM
====================================================

Backend for existing MLOps Dashboard.

Implement:

Model Registry

Training Runs

Experiment Tracking

Metrics

Versioning

Promotion

Rollback

Drift Detection

Retraining

Tables:

model_registry

training_runs

model_metrics

====================================================
ENDPOINTS
====================================================

/mlops/models

/mlops/train

/mlops/promote

/mlops/rollback

/mlops/drift

/mlops/metrics

====================================================
PHASE 9
IOT
====================================================

Implement:

ESP32

DHT22

Soil Moisture

Flow:

ESP32
↓
FastAPI
↓
SQLite
↓
Digital Twin
↓
Dashboard

Endpoints:

POST /iot/sensor-data

GET /iot/latest

====================================================
PHASE 10
REPORTING
====================================================

Generate:

Disease Report

Crop Report

Water Report

Yield Report

Digital Twin Report

Farm Summary Report

PDF Export

====================================================
PHASE 11
API STANDARDIZATION
====================================================

All APIs must return:

{
  "success": true,
  "message": "Operation completed",
  "data": {}
}

Generate:

OpenAPI

Swagger

Typed DTOs

Frontend contracts

====================================================
PHASE 12
TESTING
====================================================

Generate:

Unit Tests

Integration Tests

API Tests

ML Tests

Digital Twin Tests

MLOps Tests

Target:

90% Coverage

====================================================
PHASE 13
DEPLOYMENT
====================================================

Generate:

Dockerfile

Docker Compose

.env.example

startup scripts

README

Deployment Guide

====================================================
STRICT RULES
====================================================

DO NOT CREATE A NEW PROJECT.

MODERNIZE THE EXISTING PROJECT.

PRESERVE WORKING CODE.

REMOVE ONLY:

Broken code
Duplicate code
Dead code
Mock code

REPLACE MOCKS WITH REAL IMPLEMENTATIONS.

Use ONLY:

FastAPI
SQLite
Qwen2.5-Coder 3B
SmolVLM 3B
FAISS
XGBoost
LightGBM
CatBoost
RandomForest
IsolationForest

Build a complete production-grade AGRISENSE backend fully integrated with the existing frontend, Digital Twin, AgriGPT, IoT layer, ML engine, and MLOps dashboard.