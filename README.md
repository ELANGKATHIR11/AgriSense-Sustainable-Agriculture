# 🌾 AgriSense — Smart Agriculture & IoT Platform with AI/ML & 4D Digital Twin

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React: 19](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178c6.svg)](https://www.typescriptlang.org/)
[![PostgreSQL: 18](https://img.shields.io/badge/PostgreSQL-18.x-336791.svg)](https://www.postgresql.org/)
[![Hardware: NVIDIA RTX](https://img.shields.io/badge/CUDA-RTX%205060%20dGPU-76b900.svg)](https://www.nvidia.com/)
[![Ollama: Local LLM](https://img.shields.io/badge/Ollama-Qwen2.5%200.5B-black.svg)](https://ollama.com/)

> **AgriSense** is an enterprise, offline-capable smart agriculture platform combining real-time IoT sensor telemetry, local edge AI/ML intelligence, biophysical 4D digital twin simulation, precision automated irrigation control, and native 7-language Indian localization (English, Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali).

---

## 🏛️ System Architecture

```mermaid
flowchart TB
    subgraph IoT_Layer["Edge IoT and Sensor Network"]
        S1["Soil Moisture and Temp Sensors"]
        S2["Solar Radiation and Lux Sensors"]
        S3["Soil NPK and pH Probes"]
        S4["Drip Flow and Tank Level Sensors"]
    end

    subgraph Hardware_Gateway["Hardware Gateway and Node.js Proxy :3000"]
        GW["Express Proxy and Static Vite Server"]
    end

    subgraph Core_Backend["FastAPI Intelligence Engine :8004"]
        API["FastAPI REST and WebSocket Router"]
        TS["Live Telemetry Background Worker"]
        IM["Automated Irrigation Controller"]
        OPS["Operations and Hardware Telemetry"]
        M1["Plant Disease Vision CNN - PyTorch"]
        M2["Crop Yield Regressor - Scikit-Learn"]
        M3["Crop NPK Recommender - Random Forest"]
        M4["Digital Twin Biophysical Simulator"]
        M5["Ollama Qwen 2.5 LLM with Local RAG"]
    end

    subgraph Data_Layer["Persistence and Storage Layer"]
        PG[("PostgreSQL Database :5432")]
        LFS[("Git LFS Model Weights")]
    end

    subgraph Presentation_Layer["Reactive Frontend - React 19 and Tailwind 4"]
        UI1["Sensor Telemetry Dashboard"]
        UI2["4D Digital Twin and Three.js 3D Canvas"]
        UI3["Precision Irrigation Matrix"]
        UI4["Leaf Disease Diagnostics Studio"]
        UI5["Multilingual AI Farm Assistant"]
        UI6["Operations and MLOps Center"]
    end

    IoT_Layer -->|MQTT or HTTP Telemetry| GW
    GW <-->|API Proxy| API
    API --> TS
    API --> IM
    API --> OPS
    API --> M1
    API --> M2
    API --> M3
    API --> M4
    API --> M5
    API <-->|SQLAlchemy ORM| PG
    M1 -.->|Model Weights| LFS
    M2 -.->|Model Weights| LFS
    GW <-->|HTTP and JSON REST| Presentation_Layer
```

---

## 📦 Packages & Technologies Used

### Frontend Stack (`package.json`)
| Package | Version | Purpose |
| :--- | :--- | :--- |
| **react** | `^19.0.1` | Core declarative component rendering engine |
| **react-dom** | `^19.0.1` | React DOM bindings for web application |
| **typescript** | `^7.0.2` | Type-safety, strict interfaces, and compile-time verification |
| **vite** | `^8.3.0` | Next-generation ultra-fast frontend build tooling and HMR |
| **tailwindcss** | `^4.3.3` | Modern utility-first CSS styling and dark/light design system |
| **@tailwindcss/vite** | `^4.3.3` | Vite compilation plugin for Tailwind CSS 4 engine |
| **three** | `^0.186.1` | WebGL 3D rendering engine for Farm terrain, canopy, and sensors |
| **@types/three** | `^0.186.0` | TypeScript definitions for Three.js |
| **lucide-react** | `^0.546.0` | Complete suite of modern icons |
| **motion** | `^12.23.24` | Smooth declarative physics-based micro-animations and transitions |
| **i18next** | `^26.5.2` | Internationalization framework |
| **react-i18next** | `^17.1.0` | React hooks for 7-language Indian localization |
| **express** | `^4.21.2` | Reverse proxy and local edge server gateway |
| **tsx** | `^4.21.0` | TypeScript execution environment for server gateway |
| **dotenv** | `^17.2.3` | Environment variable loader |

### Backend Python Stack (`backend/python/requirements.txt`)
| Package | Version | Purpose |
| :--- | :--- | :--- |
| **fastapi** | `>=0.115.0` | High-performance asynchronous REST API framework |
| **uvicorn** | `>=0.32.0` | Lightning-fast ASGI web server |
| **torch** | `>=2.4.0` | PyTorch deep learning framework with CUDA dGPU support |
| **torchvision** | `>=0.19.0` | Computer vision transforms and models for plant leaf diagnostics |
| **scikit-learn** | `>=1.5.0` | Machine learning regressors, classifiers, and encoders |
| **joblib** | `>=1.4.2` | Pipeline serialization for ML models and RAG indexes |
| **pandas** | `>=2.2.0` | Tabular data analysis and dataset feature processing |
| **numpy** | `>=1.26.0` | Fast mathematical arrays and tensor calculations |
| **sqlalchemy** | `>=2.0.35` | Production ORM database abstraction layer |
| **psycopg2-binary**| `>=2.9.9` | High-performance PostgreSQL database adapter |
| **pydantic** | `>=2.9.0` | Data parsing, validation, and request/response schema modeling |
| **pydantic-settings**| `>=2.5.0` | Hierarchical environment and settings configuration |
| **python-dotenv** | `>=1.0.1` | Local configuration loading |
| **httpx** | `>=0.27.0` | Asynchronous HTTP client for Ollama LLM integration |
| **psutil** | `>=6.0.0` | System and hardware telemetry (CPU, RAM, Disk, Process IO) |
| **Pillow** | `>=10.4.0` | Image processing for leaf specimen uploads |
| **pytest** | `>=8.3.0` | Automated testing framework (100% test pass rate) |
| **pytest-asyncio**| `>=0.24.0` | Async unit test runners |

---

## 🚀 Key Modules & Capabilities

### 1. 🌐 4D Biophysical Digital Twin
- **Coupled Simulations**: FAO-56 Penman-Monteith Evapotranspiration ($ET_0$) & Growing Degree Day (GDD) crop phenology.
- **Parametric Sliders**: Real-time slider perturbation ($\Delta$ Precipitation %, Canopy Temperature Anomaly $\Delta T$, Drip Delivery Efficiency %, NPK Uptake Efficiency %).
- **Real-Time ML Invocations**: Sliders immediately invoke `crop_yield_model.joblib` via `YieldPredictor` to forecast yield changes in real-time, modifying the trajectory graphs and the WaterTwin, SoilTwin, and CropTwin cards.
- **3D Spatial Viewport**: Interactive 3D Farm visualization built with Three.js showing soil moisture heatmaps, irrigation pipelines, solar towers, and crop canopies.

### 2. 🤖 Local AI & Machine Learning Suite
- **Plant Disease Diagnosis**: PyTorch CNN trained on 38 plant disease classes (e.g., Tomato Early Blight, Corn Common Rust, Potato Late Blight, Healthy crops).
- **NPK Crop Recommendation**: Multi-class Random Forest evaluating NPK nutrient ratios, pH, rainfall, and climate against a catalog of 200 Indian crops.
- **Predictive Yield Engine**: Scikit-Learn `HistGradientBoostingRegressor` trained on 250,000 empirical agricultural observations.
- **Local RAG Advisory**: Context-aware retrieval engine indexed with 50,000 agricultural chunks, connected to local Ollama (`qwen2.5:0.5b-instruct`).

### 3. 📡 Live PostgreSQL IoT Telemetry Engine
- Background telemetry service automatically ticks live sensor readings into PostgreSQL every 4 seconds.
- Zone-specific telemetry tracking (ZONE-A, ZONE-B, ZONE-C) for moisture, temperature, EC, pH, solar lux, flow rates, and water tank levels.
- Hardware configuration endpoints allowing custom sampling intervals, sensor calibration offsets, and automated irrigation triggers.

### 4. 🇮🇳 7 Indian Languages Native Localization
- English (`en`), Hindi (`hi`), Tamil (`ta`), Telugu (`te`), Kannada (`kn`), Malayalam (`ml`), and Bengali (`bn`).
- Context-sensitive number formatting, date localizations, and agricultural glossaries.

---

## 🛠️ Quick Start & Setup

### Prerequisites
- **Node.js** `v18+` or `v20+`
- **Python** `3.11+` with PyTorch & CUDA (optional for GPU acceleration)
- **PostgreSQL** `15+` running locally on port `5432`
- **Ollama** running locally on port `11434` (with `qwen2.5:0.5b-instruct`)

### 1. Clone & Configure Environment
```bash
git clone https://github.com/ELANGKATHIR11/AgriSense-Sustainable-Agriculture.git
cd AgriSense-Sustainable-Agriculture

# Copy and adjust environment variables
cp .env.example .env
```

### 2. Set Up Database
Create the PostgreSQL database:
```sql
CREATE DATABASE agrisense;
```

### 3. Start Python ML Backend
```bash
cd backend/python
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

### 4. Start Node.js Gateway & Frontend
In the root directory:
```bash
npm install
npm run dev
```

Visit **`http://localhost:3000`** in your browser.

---

## 🧪 Testing & Verification

Run the comprehensive automated pytest suite:
```bash
python -m pytest backend/python/tests -v
```
All 11 API endpoints and ML inference pipelines pass with 100% test coverage.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
