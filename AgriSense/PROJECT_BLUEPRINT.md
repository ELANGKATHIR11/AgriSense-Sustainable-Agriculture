# 🌾 AgriSense - Complete Project Blueprint

> **A Smart Agriculture Full-Stack IoT Platform with 18+ ML Models**

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Technology Stack](#-technology-stack)
3. [Architecture Diagram](#-architecture-diagram)
4. [Directory Structure](#-directory-structure)
5. [Backend Details](#-backend-details)
6. [Frontend Details](#-frontend-details)
7. [All 18 ML Models](#-all-18-ml-models)
8. [IoT Integration](#-iot-integration)
9. [Database Schema](#-database-schema)
10. [API Endpoints](#-api-endpoints)
11. [Setup from Scratch](#-setup-from-scratch)
12. [Environment Variables](#-environment-variables)
13. [Deployment Guide](#-deployment-guide)
14. [Testing Strategy](#-testing-strategy)

---

## 🎯 Project Overview

### What is AgriSense?

AgriSense is a **comprehensive full-stack agricultural IoT platform** that combines:
- **Real-time IoT Sensor Monitoring** (ESP32 & Arduino)
- **18+ Machine Learning Models** for intelligent farming
- **AI-Powered Chatbot** with multilingual support
- **Computer Vision** for disease/weed detection
- **Weather Integration** and yield prediction
- **Smart Irrigation** recommendations

### Key Features

| Feature | Description | Technology |
|---------|-------------|------------|
| 🌡️ **Sensor Dashboard** | Real-time IoT data visualization | React + WebSocket + Three.js |
| 🤖 **AI Chatbot** | Agricultural Q&A with context | LLM + RAG + SBERT |
| 🌱 **Crop Recommendation** | ML-based crop suggestions | RandomForest + XGBoost + TensorFlow |
| 🔬 **Disease Detection** | Image-based disease identification | CNN + VLM + Transfer Learning |
| 📊 **Yield Prediction** | Harvest yield forecasting | LSTM + Regression |
| 🌿 **Weed Detection** | Automated weed identification | Image Segmentation + VLM |
| 💧 **Water Optimization** | Smart irrigation scheduling | ML + ET0 Calculation |
| 📱 **PWA Support** | Works offline on mobile | Service Workers |

---

## 🛠️ Technology Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.12.10 | Core language |
| FastAPI | 0.115.6+ | REST API framework |
| Uvicorn | 0.34.0+ | ASGI server |
| SQLAlchemy | 2.0.36+ | ORM |
| SQLite | 3.x | Development database |
| Redis | 5.2.1+ | Caching & Celery broker |
| Celery | 5.4.0+ | Background tasks |

### Machine Learning
| Technology | Version | Purpose |
|------------|---------|---------|
| TensorFlow | 2.18+ | Deep learning |
| PyTorch | 2.5+ | Vision models |
| scikit-learn | 1.6.1+ | Classical ML |
| Transformers | 4.47+ | NLP & VLM |
| OpenCV | 4.10+ | Computer vision |
| sentence-transformers | latest | Embeddings |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.3+ | UI framework |
| TypeScript | 5.x | Type safety |
| Vite | 5.x | Build tool |
| TailwindCSS | 3.x | Styling |
| Three.js | latest | 3D visualizations |
| React Query | 5.x | Data fetching |
| i18next | 25.x | Internationalization |

### IoT
| Technology | Purpose |
|------------|---------|
| ESP32 | WiFi-enabled sensor hub |
| Arduino Nano | Temperature sensing |
| DHT22 | Temperature & humidity |
| DS18B20 | Soil temperature |
| Capacitive | Soil moisture |
| MQTT | IoT communication |

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AgriSense Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   ESP32     │     │  Arduino    │     │  Weather    │                   │
│  │   Sensors   │     │   Nano      │     │    API      │                   │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │
│         │                   │                   │                           │
│         └───────────┬───────┴───────────────────┘                           │
│                     │ MQTT / Serial / HTTP                                  │
│                     ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        FastAPI Backend                               │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │ Sensor  │ │   ML    │ │ Chatbot │ │Disease  │ │  Weed   │       │   │
│  │  │   API   │ │ Models  │ │   RAG   │ │Detection│ │Detection│       │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │   │
│  │       │           │           │           │           │             │   │
│  │  ┌────┴───────────┴───────────┴───────────┴───────────┴────┐       │   │
│  │  │                    Core Engine                           │       │   │
│  │  │  - RecoEngine (Rule-based recommendations)              │       │   │
│  │  │  - SmartFarmingML (18+ ML Models)                       │       │   │
│  │  │  - VLM Engine (Vision-Language Models)                  │       │   │
│  │  │  - PlantHealthMonitor (Comprehensive analysis)          │       │   │
│  │  └─────────────────────────────────────────────────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                     │                                                       │
│                     ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    React Frontend (PWA)                              │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │Dashboard│ │ Chatbot │ │ Disease │ │  Crops  │ │Irrigation│       │   │
│  │  │  3D UI  │ │   UI    │ │ Scanner │ │ Manager │ │ Control │       │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
AgriSense/
├── agrisense_app/
│   ├── backend/                    # FastAPI Backend
│   │   ├── main.py                 # Application entry point
│   │   ├── api/                    # API routes
│   │   │   ├── ai_routes.py        # AI/ML endpoints
│   │   │   ├── sensor_api.py       # Sensor data endpoints
│   │   │   └── mqtt_sensor_bridge.py
│   │   ├── core/                   # Core business logic
│   │   │   ├── engine.py           # RecoEngine - recommendation engine
│   │   │   ├── data_store.py       # Database operations
│   │   │   └── config.yaml         # Crop parameters
│   │   ├── routes/                 # Additional routes
│   │   │   ├── ai_models_routes.py # Phi LLM & SCOLD VLM endpoints
│   │   │   ├── vlm_routes.py       # Vision-Language Model routes
│   │   │   ├── ml_predictions.py   # ML prediction endpoints
│   │   │   └── health_routes.py    # Health check endpoints
│   │   ├── models/                 # Trained ML Models (18+)
│   │   │   ├── crop_recommendation_rf.joblib
│   │   │   ├── crop_recommendation_gb.joblib
│   │   │   ├── yield_prediction_model.joblib
│   │   │   ├── water_model.joblib
│   │   │   ├── disease_model_latest.joblib
│   │   │   ├── weed_model_latest.joblib
│   │   │   ├── intent_classifier.joblib
│   │   │   ├── crop_recommendation_tf_medium/
│   │   │   └── ... (more models)
│   │   ├── nlp/                    # NLP modules
│   │   │   └── response_generator.py
│   │   ├── ml/                     # ML utilities
│   │   │   ├── inference_optimized.py
│   │   │   └── model_optimizer.py
│   │   ├── vlm/                    # Vision-Language Models
│   │   │   └── vlm_service.py
│   │   ├── trainers/               # Model training scripts
│   │   │   └── runner.py
│   │   ├── smart_farming_ml.py     # SmartFarmingRecommendationSystem
│   │   ├── disease_detection.py    # DiseaseDetectionEngine
│   │   ├── weed_management.py      # WeedManagementEngine
│   │   ├── smart_weed_detector.py  # SmartWeedDetector
│   │   ├── plant_health_monitor.py # PlantHealthMonitor
│   │   ├── chatbot_conversational.py # ConversationalEnhancer
│   │   ├── hybrid_agri_ai.py       # HybridAgriAI (LLM+VLM)
│   │   ├── vlm_engine.py           # AgriVLMEngine
│   │   └── requirements.txt        # Python dependencies
│   │
│   └── frontend/
│       └── farm-fortune-frontend-main/
│           ├── src/
│           │   ├── pages/          # Page components
│           │   │   ├── Dashboard.tsx
│           │   │   ├── Chatbot.tsx
│           │   │   ├── DiseaseManagement.tsx
│           │   │   ├── WeedManagement.tsx
│           │   │   ├── Crops.tsx
│           │   │   ├── Irrigation.tsx
│           │   │   ├── SoilAnalysis.tsx
│           │   │   ├── Harvesting.tsx
│           │   │   └── Arduino.tsx
│           │   ├── components/     # Reusable components
│           │   │   ├── 3d/         # Three.js 3D components
│           │   │   ├── charts/     # Chart components
│           │   │   ├── dashboard/  # Dashboard widgets
│           │   │   └── ui/         # shadcn/ui components
│           │   ├── services/       # API services
│           │   ├── hooks/          # Custom React hooks
│           │   ├── locales/        # i18n translations
│           │   │   ├── en.json
│           │   │   ├── hi.json
│           │   │   ├── ta.json
│           │   │   └── te.json
│           │   └── lib/            # Utilities
│           ├── package.json
│           └── vite.config.ts
│
├── AGRISENSE_IoT/                  # IoT Firmware
│   ├── esp32_firmware/
│   │   ├── agrisense_esp32.ino     # ESP32 main firmware
│   │   ├── platformio.ini
│   │   └── SETUP_GUIDE.md
│   ├── arduino_nano_firmware/
│   │   ├── agrisense_nano_temp_sensor.ino
│   │   ├── arduino_bridge.py       # Serial-to-backend bridge
│   │   └── WIRING_GUIDE.md
│   └── esp32_config.py
│
├── datasets/                       # Training datasets
│   ├── chatbot/
│   │   └── merged_chatbot_training_dataset.csv
│   ├── enhanced/
│   └── raw/
│
├── documentation/                  # Project documentation
│   ├── developer/
│   ├── architecture/
│   └── guides/
│
├── tests/                          # Test suites
├── scripts/                        # Utility scripts
└── tools/                          # Development tools
```

---

## 🔧 Backend Details

### Main Application (`main.py`)

The FastAPI application provides:
- **RESTful API** endpoints for all features
- **WebSocket** support for real-time sensor data
- **CORS** configuration for frontend communication
- **GZip** compression middleware
- **Static file** serving for frontend assets

### Core Modules

#### 1. RecoEngine (`core/engine.py`)
```python
class RecoEngine:
    """
    Rule-based recommendation engine for irrigation and fertilization.
    
    Features:
    - ET0 (Evapotranspiration) calculation using Hargreaves method
    - Crop-specific water requirements (Kc coefficients)
    - NPK fertilizer recommendations
    - Soil type adjustments
    - Cost and CO2 impact calculations
    """
```

#### 2. SmartFarmingRecommendationSystem (`smart_farming_ml.py`)
```python
class SmartFarmingRecommendationSystem:
    """
    ML-powered crop recommendation system.
    
    Models Used:
    - RandomForest for yield prediction
    - RandomForest for crop classification
    - TensorFlow models for enhanced predictions
    
    Input Features:
    - pH, N, P, K levels
    - Temperature, humidity, moisture
    - Soil type
    """
```

#### 3. DiseaseDetectionEngine (`disease_detection.py`)
```python
class DiseaseDetectionEngine:
    """
    Plant disease detection using computer vision.
    
    Capabilities:
    - Image preprocessing (PIL + OpenCV)
    - CNN-based disease classification
    - VLM integration for enhanced analysis
    - Treatment recommendations
    """
```

#### 4. WeedManagementEngine (`weed_management.py`)
```python
class WeedManagementEngine:
    """
    Weed detection and management recommendations.
    
    Features:
    - Image segmentation for weed identification
    - Coverage analysis
    - Control method recommendations
    - Economic impact assessment
    """
```

#### 5. ConversationalEnhancer (`chatbot_conversational.py`)
```python
class ConversationalEnhancer:
    """
    Makes chatbot responses farmer-friendly.
    
    Features:
    - Multi-turn conversation memory
    - Empathetic language
    - Multilingual support (EN, HI, TA, TE, KN)
    - Follow-up suggestions
    """
```

#### 6. HybridAgriAI (`hybrid_agri_ai.py`)
```python
class HybridAgriAI:
    """
    Hybrid LLM+VLM Edge AI system.
    
    Components:
    - Phi LLM (Ollama) for text understanding
    - SCOLD VLM for visual analysis
    - Offline-first architecture
    - Multimodal analysis (image + text)
    """
```

---

## 🎨 Frontend Details

### Tech Stack
- **React 18.3** with TypeScript
- **Vite** for fast development
- **TailwindCSS** for styling
- **shadcn/ui** component library
- **Three.js** for 3D visualizations
- **React Query** for data fetching
- **i18next** for internationalization

### Pages

| Page | Route | Description |
|------|-------|-------------|
| Home | `/` | Landing page with features |
| Dashboard | `/dashboard` | Real-time sensor monitoring |
| Chatbot | `/chatbot` | AI agricultural assistant |
| Disease | `/disease` | Disease detection & management |
| Weeds | `/weeds` | Weed identification |
| Crops | `/crops` | Crop recommendation |
| Irrigation | `/irrigation` | Water management |
| Soil | `/soil` | Soil analysis |
| Harvesting | `/harvesting` | Harvest predictions |
| Arduino | `/arduino` | Serial sensor connection |
| Tank | `/tank` | Water tank monitoring |
| Admin | `/admin` | System administration |

### Key Components

```
components/
├── 3d/                     # Three.js 3D visualizations
│   ├── SensorVisualization.tsx
│   └── CropGrowthModel.tsx
├── charts/                 # Data visualization
│   ├── SensorChart.tsx
│   ├── YieldChart.tsx
│   └── ComparisonChart.tsx
├── dashboard/              # Dashboard widgets
│   ├── SensorCard.tsx
│   ├── AlertPanel.tsx
│   └── RecommendationPanel.tsx
├── ui/                     # shadcn/ui components
│   ├── button.tsx
│   ├── card.tsx
│   └── ... (50+ components)
├── CropDetector.tsx        # Image upload & analysis
├── ArduinoSerialConnection.tsx  # Web Serial API
└── LanguageSwitcher.tsx    # i18n language selector
```

### Internationalization

Supported languages:
- 🇬🇧 English (en)
- 🇮🇳 Hindi (hi)
- 🇮🇳 Tamil (ta)
- 🇮🇳 Telugu (te)
- 🇮🇳 Kannada (kn)

---

## 🤖 All 18 ML Models

### Model Overview

| # | Model Name | Type | Purpose | Input | Output |
|---|------------|------|---------|-------|--------|
| 1 | **crop_recommendation_rf** | RandomForest | Crop recommendation | Soil + Climate data | Best crop |
| 2 | **crop_recommendation_gb** | GradientBoosting | Crop recommendation | Soil + Climate data | Best crop |
| 3 | **crop_recommendation_nn** | Neural Network | Crop recommendation | Soil + Climate data | Best crop + confidence |
| 4 | **crop_recommendation_tf_small** | TensorFlow DNN | Crop classification | 8 features | Crop label |
| 5 | **crop_recommendation_tf_medium** | TensorFlow DNN | Crop classification | 8 features | Crop + probability |
| 6 | **yield_prediction_model** | RandomForest Regressor | Yield forecasting | Crop + conditions | Tonnes/hectare |
| 7 | **water_model** | RandomForest | Water optimization | Crop + weather | Liters needed |
| 8 | **fertilizer_model** | ML Regressor | NPK recommendation | Soil test + crop | NPK kg/ha |
| 9 | **disease_model_latest** | CNN + Transfer | Disease detection | Plant image | Disease + treatment |
| 10 | **weed_model_latest** | Segmentation | Weed detection | Field image | Weed type + location |
| 11 | **intent_classifier** | SVM/LogReg | Chatbot intent | User query | Intent category |
| 12 | **chatbot_encoder** | SBERT | Semantic search | Question | Embedding vector |
| 13 | **gradient_boosting_optimized** | GradientBoosting | Enhanced crop rec | Multi-feature | Crop ranking |
| 14 | **random_forest_optimized** | RandomForest | Enhanced crop rec | Multi-feature | Crop ranking |
| 15 | **disease_model_enhanced** | Joblib | Enhanced disease | Image features | Disease + confidence |
| 16 | **weed_model_enhanced** | Joblib | Enhanced weed | Image features | Weed + severity |
| 17 | **crop_recommendation_tf_npu** | TensorFlow NPU | NPU-optimized | 8 features | Crop label |
| 18 | **openvino_npu_models** | OpenVINO | Intel NPU inference | Various | Accelerated inference |

### Detailed Model Descriptions

#### 1-3. Crop Recommendation Models (Classical ML)

```python
# Location: models/crop_recommendation_rf.joblib
# Training: SmartFarmingRecommendationSystem.prepare_models()

Input Features (8):
- pH_Optimal
- Nitrogen_Optimal_kg_ha
- Phosphorus_Optimal_kg_ha
- Potassium_Optimal_kg_ha
- Temperature_Optimal_C
- Water_Requirement_mm
- Moisture_Optimal_percent
- Humidity_Optimal_percent
- Soil_Type_Encoded

Output: Crop name from 48+ Indian crops
Accuracy: ~92%
```

#### 4-5. TensorFlow Crop Models

```python
# Location: models/crop_recommendation_tf_medium/
# Architecture: Dense Neural Network

Layers:
- Input: 8 features
- Dense(128, relu)
- Dropout(0.3)
- Dense(64, relu)
- Dropout(0.2)
- Dense(num_crops, softmax)

Training Data: india_crop_dataset.csv (1000+ samples)
Accuracy: ~94%
```

#### 6. Yield Prediction Model

```python
# Location: models/yield_prediction_model.joblib
# Type: RandomForest Regressor

Features:
- Crop type (encoded)
- Soil conditions
- Climate parameters
- Historical yield data

Output: Expected yield in tonnes/hectare
MAE: ~0.3 tonnes/ha
```

#### 7. Water Optimization Model

```python
# Location: models/water_model.joblib
# Combines ML with ET0 (Evapotranspiration) calculations

Hargreaves ET0 Formula:
ET0 = 0.0023 * Ra * (Tavg + 17.8) * sqrt(Tmax - Tmin)

Features:
- Crop Kc coefficient
- Temperature data
- Soil type
- Growth stage

Output: Daily water requirement (mm/day)
```

#### 8. Fertilizer Recommendation Model

```python
# Location: models/fertilizer_recommendation_model.joblib

Input:
- Current soil NPK levels
- Target crop requirements
- Soil pH
- Organic matter %

Output:
- Nitrogen (kg/ha)
- Phosphorus (kg/ha)
- Potassium (kg/ha)
```

#### 9. Disease Detection Model

```python
# Location: models/disease_model_latest.joblib
# Class: DiseaseDetectionEngine

Supported Diseases (20+):
- Leaf spot, Powdery mildew, Rust
- Blight (early/late), Mosaic virus
- Bacterial wilt, Root rot
- Anthracnose, Downy mildew
- Various nutrient deficiencies

Pipeline:
1. Image preprocessing (resize to 224x224)
2. Feature extraction (CNN/ResNet)
3. Classification
4. Treatment recommendation lookup
```

#### 10. Weed Detection Model

```python
# Location: models/weed_model_latest.joblib
# Class: WeedManagementEngine

Supported Weeds (15+):
- Dandelion, Crabgrass, Clover
- Plantain, Chickweed, Pigweed
- Lambsquarters, Foxtail, Bindweed

Output:
- Weed type
- Coverage percentage
- Location in field
- Control recommendations
```

#### 11. Intent Classifier

```python
# Location: models/intent_classifier.joblib
# Used by: Chatbot for query routing

Intent Categories:
- crop_recommendation
- disease_query
- weed_management
- irrigation_advice
- fertilizer_query
- weather_query
- pest_control
- general_farming
- greeting
- farewell
```

#### 12. Chatbot Encoder (SBERT)

```python
# Used for: Semantic similarity in RAG pipeline

Model: all-MiniLM-L6-v2
Embedding Size: 384 dimensions
Purpose: Match user queries to knowledge base

Pipeline:
1. Encode user question
2. Find similar Q&A pairs (cosine similarity)
3. Retrieve top-k matches
4. Generate contextual response
```

### Model Training

```bash
# Train all models
python -m agrisense_app.backend.trainers.runner

# Train specific model
python scripts/train_crop_models.py
python scripts/train_disease_model.py
python scripts/train_weed_model.py
python scripts/train_chatbot.py
```

### Model Performance Summary

| Model Category | Accuracy | F1-Score | Notes |
|----------------|----------|----------|-------|
| Crop Recommendation | 92-94% | 0.91 | 48 crop classes |
| Disease Detection | 87-90% | 0.86 | 20+ diseases |
| Weed Detection | 85-88% | 0.84 | 15+ weed types |
| Intent Classification | 95%+ | 0.94 | 10 intents |
| Yield Prediction | MAE: 0.3 | R²: 0.89 | Regression |

---

## 📡 IoT Integration

### ESP32 Sensor Hub

```cpp
// File: AGRISENSE_IoT/esp32_firmware/agrisense_esp32.ino

Supported Sensors:
├── DHT22           → Air temperature & humidity
├── DS18B20         → Soil temperature
├── Capacitive      → Soil moisture
├── pH Probe        → Soil pH level
└── LDR             → Light intensity

Communication: MQTT over WiFi
Data Format: JSON
Interval: 30 seconds
```

**Wiring Diagram:**
```
ESP32 Pin Connections:
┌─────────┬─────────────────┐
│ GPIO 2  │ DHT22 Data      │
│ GPIO 34 │ Soil Moisture   │
│ GPIO 4  │ DS18B20 Data    │
│ GPIO 35 │ pH Sensor       │
│ GPIO 36 │ Light Sensor    │
│ GPIO 5  │ Pump Relay      │
└─────────┴─────────────────┘
```

### Arduino Nano Temperature Module

```cpp
// File: AGRISENSE_IoT/arduino_nano_firmware/agrisense_nano_temp_sensor.ino

Sensors:
├── DS18B20 (Pin 2) → Soil temperature
└── DHT22 (Pin 3)   → Air temp & humidity

Communication: Serial (115200 baud)
Data Format: JSON
Interval: 5 seconds
```

### Data Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  ESP32   │────▶│   MQTT   │────▶│ FastAPI  │────▶│ Database │
│ Sensors  │     │  Broker  │     │ Backend  │     │ (SQLite) │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                                        │
┌──────────┐     ┌──────────┐           │
│ Arduino  │────▶│  Serial  │───────────┘
│   Nano   │     │  Bridge  │
└──────────┘     └──────────┘
```

---

## 💾 Database Schema

### SQLite Tables

```sql
-- Sensor Readings
CREATE TABLE sensor_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    temperature REAL,
    humidity REAL,
    soil_moisture REAL,
    soil_temperature REAL,
    ph_level REAL,
    light_intensity REAL,
    nitrogen REAL,
    phosphorus REAL,
    potassium REAL
);

-- Tank Level
CREATE TABLE tank_level (
    id INTEGER PRIMARY KEY,
    level_percentage REAL,
    volume_liters REAL,
    last_updated DATETIME
);

-- Irrigation Log
CREATE TABLE irrigation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    duration_minutes INTEGER,
    water_volume_liters REAL,
    triggered_by TEXT,  -- 'auto' or 'manual'
    zone_id TEXT
);

-- Alert Log
CREATE TABLE alert_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    alert_type TEXT,
    severity TEXT,  -- 'low', 'medium', 'high', 'critical'
    message TEXT,
    acknowledged BOOLEAN DEFAULT 0
);

-- Recommendation Log
CREATE TABLE reco_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    crop_type TEXT,
    recommendation_type TEXT,
    details TEXT,
    confidence REAL
);

-- Weather Cache
CREATE TABLE weather_cache (
    id INTEGER PRIMARY KEY,
    location TEXT,
    data TEXT,  -- JSON
    fetched_at DATETIME
);
```

---

## 🔌 API Endpoints

### Sensor Endpoints
```
GET  /api/sensors/readings        # Get all sensor readings
POST /api/sensors/readings        # Add new sensor reading
GET  /api/sensors/latest          # Get latest readings
GET  /api/tank/level              # Get tank water level
POST /api/tank/level              # Update tank level
```

### ML Prediction Endpoints
```
POST /api/recommend               # Get crop recommendation
POST /api/yield/predict           # Predict crop yield
POST /api/water/optimize          # Get water optimization
POST /api/fertilizer/recommend    # Get NPK recommendation
```

### Disease & Weed Detection
```
POST /api/disease/detect          # Detect plant disease
POST /api/disease/detect-scold    # VLM-powered detection
POST /api/weed/detect             # Detect weeds
POST /api/weed/detect-scold       # VLM-powered weed detection
POST /api/plant-health/assess     # Comprehensive health check
```

### Chatbot Endpoints
```
POST /api/chat                    # Send message to chatbot
POST /api/chatbot/enrich          # Enrich answer with LLM
POST /api/chatbot/contextual      # Contextual response
GET  /api/chatbot/history/{id}    # Get conversation history
```

### System Endpoints
```
GET  /health                      # Health check
GET  /api/models/status           # ML models status
GET  /api/phi/status              # Phi LLM status
GET  /api/scold/status            # SCOLD VLM status
```

---

## 🚀 Setup from Scratch

### Prerequisites

```bash
# System requirements
- Python 3.12.10
- Node.js 20.x LTS
- Git
- MQTT Broker (Mosquitto) - optional
- Redis - optional (for Celery)
```

### Step 1: Clone Repository

```bash
git clone https://github.com/ELANGKATHIR11/AgriSense-A-Smart-Agriculture-Solution.git
cd AgriSense-A-Smart-Agriculture-Solution
```

### Step 2: Backend Setup

```bash
# Navigate to backend
cd agrisense_app/backend

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For ML features (optional, heavy)
pip install -r requirements-ml.txt

# Initialize database
python -c "from core.data_store import init_sensor_db; init_sensor_db()"

# Start backend server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Frontend Setup

```bash
# Navigate to frontend
cd agrisense_app/frontend/farm-fortune-frontend-main

# Install dependencies
npm install

# Start development server
npm run dev
```

### Step 4: IoT Setup (Optional)

```bash
# ESP32
1. Install PlatformIO or Arduino IDE
2. Open AGRISENSE_IoT/esp32_firmware/agrisense_esp32.ino
3. Update WiFi credentials in code
4. Update MQTT broker IP
5. Flash to ESP32

# Arduino Nano
1. Open AGRISENSE_IoT/arduino_nano_firmware/agrisense_nano_temp_sensor.ino
2. Flash to Arduino Nano
3. Start serial bridge:
   python arduino_bridge.py --port COM3
```

### Step 5: Train ML Models (Optional)

```bash
# Train all models
cd agrisense_app/backend
python -m trainers.runner

# Or train specific models
python scripts/train_crop_models.py
python scripts/train_disease_model.py
```

### Quick Start Commands

```bash
# One-liner backend start
cd agrisense_app/backend && python -m uvicorn main:app --reload

# One-liner frontend start
cd agrisense_app/frontend/farm-fortune-frontend-main && npm run dev

# Both (separate terminals)
# Terminal 1: Backend at http://localhost:8000
# Terminal 2: Frontend at http://localhost:5173
```

---

## ⚙️ Environment Variables

### Backend (.env)

```env
# Core Settings
AGRISENSE_DISABLE_ML=0          # Set to 1 to disable ML models
DEBUG=true
LOG_LEVEL=INFO

# Database
DATABASE_URL=sqlite:///./sensors.db

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# MQTT (for IoT)
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_TOPIC_PREFIX=agrisense/

# LLM Integration (Optional)
PHI_LLM_ENDPOINT=http://localhost:11434
PHI_MODEL_NAME=phi:latest
PHI_CHAT_TEMPERATURE=0.75

# VLM Integration (Optional)
SCOLD_BASE_URL=http://localhost:8001
SCOLD_CONFIDENCE_THRESHOLD=0.6

# External APIs
OPENWEATHER_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Frontend (.env)

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
VITE_ENABLE_3D=true
VITE_DEFAULT_LANGUAGE=en
```

---

## 🐳 Deployment Guide

### Docker Deployment

```dockerfile
# Backend Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY agrisense_app/backend/ .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: ./agrisense_app/backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///./sensors.db
    volumes:
      - ./data:/app/data

  frontend:
    build: ./agrisense_app/frontend/farm-fortune-frontend-main
    ports:
      - "5173:80"
    depends_on:
      - backend

  mqtt:
    image: eclipse-mosquitto:latest
    ports:
      - "1883:1883"

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### Azure Deployment

```bash
# Azure Container Apps
az containerapp up \
  --name agrisense-backend \
  --resource-group agrisense-rg \
  --source ./agrisense_app/backend \
  --ingress external \
  --target-port 8000
```

---

## 🧪 Testing Strategy

### Backend Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=agrisense_app --cov-report=html

# Run specific test
pytest tests/test_ml_outputs.py -v
```

### Frontend Tests

```bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e

# Coverage
npm run test:coverage
```

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# Crop recommendation
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"temperature": 25, "humidity": 60, "ph": 6.5, "nitrogen": 40}'
```

---

## 📞 Support & Contact

- **Repository**: https://github.com/ELANGKATHIR11/AgriSense-A-Smart-Agriculture-Solution
- **Author**: ELANGKATHIR11
- **Version**: 1.0.0
- **License**: MIT

---

## 🙏 Acknowledgments

- **TensorFlow** & **PyTorch** for ML frameworks
- **FastAPI** for the excellent web framework
- **React** & **Three.js** for beautiful UI
- **shadcn/ui** for component library
- **Hugging Face** for pre-trained models

---

**🌾 Happy Farming with AgriSense! 🚜**
