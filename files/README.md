# 🌾 AgriSense - Smart Agriculture Full-Stack IoT Platform

> **A comprehensive agricultural IoT platform with 18+ ML models, real-time monitoring, and AI-powered insights**

![Version](https://img.shields.io/badge/version-2.0.0-green.svg)
![Python](https://img.shields.io/badge/python-3.12.10-blue.svg)
![React](https://img.shields.io/badge/react-18.3-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Backend Setup](#-backend-setup)
- [Frontend Setup](#-frontend-setup)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### Core Capabilities

| Feature | Description | Technology |
|---------|-------------|------------|
| 🌡️ **Real-time IoT Monitoring** | Live sensor data from ESP32 & Arduino | WebSocket + MQTT |
| 🤖 **AI Chatbot** | Multilingual agricultural assistant | FastAPI + NLP |
| 🌱 **Crop Recommendation** | ML-based crop suggestions | RandomForest + TensorFlow |
| 🔬 **Disease Detection** | Image-based plant disease ID | CNN + Computer Vision |
| 🌿 **Weed Management** | Automated weed detection | Image Segmentation |
| 📊 **Yield Prediction** | Harvest forecasting | LSTM + Regression |
| 💧 **Smart Irrigation** | ET0-based water optimization | Hargreaves Method |
| 📱 **PWA Support** | Works offline on mobile | Service Workers |

### ML Models (18+)

1. **Crop Recommendation** - RandomForest, GradientBoosting, Neural Networks
2. **Yield Prediction** - RandomForest Regressor
3. **Water Optimization** - ET0 Calculation + ML
4. **Fertilizer Recommendation** - NPK Calculation
5. **Disease Detection** - CNN + Transfer Learning
6. **Weed Detection** - Image Segmentation
7. **Chatbot Intent Classifier** - SVM/LogReg
8. **Semantic Search** - SBERT Embeddings
9. And 10 more specialized models...

---

## 🛠️ Technology Stack

### Backend
- **Python 3.12.10** - Core language
- **FastAPI 0.115.6+** - REST API framework
- **SQLite** - Database (with MongoDB support)
- **TensorFlow 2.18+** - Deep learning
- **scikit-learn 1.6.1+** - Classical ML
- **Redis** - Caching & Celery broker

### Frontend
- **React 18.3** + TypeScript
- **Vite 5.x** - Build tool
- **TailwindCSS 3.x** - Styling
- **Three.js** - 3D visualizations
- **React Query 5.x** - Data fetching
- **i18next** - Multilingual support (5 languages)

### IoT
- **ESP32** - WiFi-enabled sensor hub
- **Arduino Nano** - Temperature module
- **DHT22, DS18B20** - Sensors
- **MQTT** - Communication protocol

---

## 🚀 Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.12.10
- Node.js 20.x LTS
- Git
```

### One-Command Setup

```bash
# Clone repository
git clone <repository-url>
cd AgriSense

# Backend setup
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend setup (new terminal)
cd ../frontend
npm install
npm run dev
```

### Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Admin Dashboard**: http://localhost:5173/admin

---

## 🔧 Backend Setup

### Step 1: Create Virtual Environment

```bash
cd backend
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Core dependencies (lightweight)
pip install -r requirements.txt

# Optional ML dependencies (heavy - ~2GB)
# pip install tensorflow==2.18.0 torch==2.5.1 transformers==4.47.1
```

### Step 3: Initialize Database

```bash
# Database is auto-initialized on first run
python -c "from core.data_store import init_sensor_db; init_sensor_db()"
```

### Step 4: Start Backend Server

```bash
# Development mode with hot reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Core Settings
AGRISENSE_DISABLE_ML=0          # Set to 1 to disable ML models
DEBUG=true
LOG_LEVEL=INFO

# Database
DATABASE_URL=sqlite:///./sensors.db

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# MQTT (optional)
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883

# API Keys (optional)
OPENWEATHER_API_KEY=your_key_here
```

---

## 🎨 Frontend Setup

### Step 1: Install Dependencies

```bash
cd frontend
npm install
```

### Step 2: Configure Environment

Create a `.env` file in the frontend directory:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
VITE_ENABLE_3D=true
VITE_DEFAULT_LANGUAGE=en
```

### Step 3: Start Development Server

```bash
# Development mode with HMR
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## 📁 Project Structure

```
AgriSense/
├── backend/                     # FastAPI Backend
│   ├── main.py                  # Application entry point
│   ├── api/                     # API routes
│   │   ├── sensor_api.py        # Sensor endpoints
│   │   └── ai_routes.py         # AI/ML endpoints
│   ├── core/                    # Core business logic
│   │   ├── engine.py            # RecoEngine
│   │   ├── data_store.py        # Database operations
│   │   └── config.yaml          # Crop parameters
│   ├── routes/                  # Additional routes
│   │   ├── ml_predictions.py    # ML predictions
│   │   ├── health_routes.py     # Health checks
│   │   └── admin_routes.py      # Admin endpoints
│   ├── requirements.txt         # Python dependencies
│   └── sensors.db               # SQLite database
│
├── frontend/                    # React Frontend
│   ├── src/
│   │   ├── pages/               # Page components
│   │   │   ├── Home.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Chatbot.tsx
│   │   │   ├── Crops.tsx
│   │   │   ├── DiseaseManagement.tsx
│   │   │   ├── WeedManagement.tsx
│   │   │   ├── Irrigation.tsx
│   │   │   └── Admin.tsx
│   │   ├── components/          # Reusable components
│   │   │   └── Layout.tsx
│   │   ├── services/            # API services
│   │   │   └── api.ts
│   │   ├── App.tsx              # Main App component
│   │   ├── main.tsx             # Entry point
│   │   └── i18n.ts              # Internationalization
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── tsconfig.json
│
└── README.md                    # This file
```

---

## 🔌 API Documentation

### Health & System

```bash
GET  /health                     # Basic health check
GET  /status                     # Detailed system status
GET  /version                    # API version info
```

### Sensor Endpoints

```bash
GET  /sensors/live               # Get latest sensor data
GET  /sensors/history            # Get sensor history
GET  /sensors/devices            # List all devices
GET  /sensors/devices/status     # Device status summary
POST /sensors/data               # Post sensor reading
```

### ML Predictions

```bash
POST /recommend                  # Crop recommendation
POST /water/optimize             # Water optimization
POST /fertilizer/recommend       # Fertilizer recommendation
POST /yield/predict              # Yield prediction
GET  /crops                      # Get supported crops
GET  /models/status              # ML models status
```

### AI Services

```bash
POST /ai/chat                    # Chat with AI assistant
POST /ai/disease/detect          # Detect plant disease
POST /ai/weed/detect             # Detect weeds
POST /ai/plant-health/assess     # Plant health assessment
```

### Admin Endpoints

```bash
GET  /admin/metrics              # System metrics
GET  /admin/summary              # Dashboard summary
GET  /admin/activities           # Activity log
POST /admin/action               # Perform admin action
POST /admin/reset                # Reset system data
```

### Example API Calls

```bash
# Crop recommendation
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 25,
    "humidity": 60,
    "ph": 6.5,
    "rainfall": 100
  }'

# Get live sensor data
curl http://localhost:8000/sensors/live

# Water optimization
curl -X POST http://localhost:8000/water/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "Rice",
    "growth_stage": "mid",
    "temp_min": 20,
    "temp_max": 30,
    "soil_type": "loam"
  }'
```

---

## 🧪 Testing

### Backend Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

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

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 📊 Performance

- **Backend Response Time**: < 100ms (cached)
- **ML Inference**: < 500ms per prediction
- **Real-time Updates**: 3-second refresh interval
- **Concurrent Users**: 100+ supported
- **Database**: 10,000+ sensor readings/day

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support & Contact

- **Issues**: Please use GitHub Issues for bug reports
- **Documentation**: Check `/docs` folder for detailed guides
- **Version**: 2.0.0
- **Last Updated**: January 27, 2026

---

## 🙏 Acknowledgments

- **TensorFlow** & **PyTorch** for ML frameworks
- **FastAPI** for the excellent web framework
- **React** & **Three.js** for beautiful UI
- **Hugging Face** for pre-trained models
- Open-source community for inspiration

---

**🌾 Happy Farming with AgriSense! 🚜**

Built with ❤️ for modern agriculture
