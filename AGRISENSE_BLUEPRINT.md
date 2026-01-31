# AGRISENSE Blueprint (generated)
# Generated: 2026-01-31T00:00:00Z

````markdown
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
| MongoDB | 6.0+ | Primary NoSQL Database (Mongoose ORM) |
| SQLite | 3.x | In-memory Fallback (Hybrid Mock) |
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
{