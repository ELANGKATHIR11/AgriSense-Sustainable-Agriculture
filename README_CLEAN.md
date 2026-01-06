# AgriSense - Smart Agriculture Platform

<div align="center">
  
  **🌾 Transform Agriculture with AI, IoT & Precision Farming 🚀**
  
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.123+-teal.svg)](https://fastapi.tiangolo.com)
  [![React](https://img.shields.io/badge/React-18+-cyan.svg)](https://reactjs.org)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue.svg)](https://typescriptlang.org)
  
</div>

---

## 📖 Table of Contents

- [About](#about-agrisense)
- [Features](#key-features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## About AgriSense

AgriSense is a **production-ready smart agriculture platform** that leverages **AI, IoT, and Cloud Technologies** to optimize farming operations, increase yields, and promote sustainable agriculture.

**Perfect for:**
- 🌾 Individual farmers & farm operators
- 🏢 Agricultural enterprises & cooperatives
- 🔬 Agricultural research institutions
- 🌍 Sustainable farming initiatives

### 🎯 Core Objectives

✅ **Increase Crop Yield** - ML-driven crop selection & growth monitoring  
✅ **Optimize Water Usage** - Smart irrigation scheduling with 30-40% water savings  
✅ **Reduce Input Costs** - Precision NPK recommendations save fertilizer costs  
✅ **Detect Issues Early** - AI disease & weed detection for timely intervention  
✅ **Empower Farmers** - Intelligent chatbot in local languages (Hindi, Tamil, Telugu, Kannada)  

---

## Key Features

### 🤖 AI & Machine Learning

- **18 Advanced ML Models**
  - Crop Recommendation (Random Forest, Gradient Boost, Neural Networks)
  - Yield Prediction & Forecasting
  - Water Optimization (ET0-based calculations)
  - Fertilizer NPK Recommendations
  - Plant Disease Detection (20+ diseases)
  - Weed Identification (15+ weed types)
  - Intent Classification for Chatbot

### 📡 Real-Time IoT Integration

- **Sensor Hub** - ESP32 with DHT22, pH probe, soil moisture, light sensors
- **Temperature Module** - Arduino Nano for precise soil temperature
- **Live Data Dashboard** - Real-time metrics updated every 3 seconds
- **MQTT Support** - Flexible data ingestion pipeline

### 🎨 Modern Web Interface

- **React + TypeScript + Vite** - Fast, responsive UI
- **Admin Dashboard** - Real-time system monitoring & controls
- **Multilingual Support** - English, Hindi, Tamil, Telugu, Kannada
- **Mobile-Responsive** - Works seamlessly on all devices
- **3D Visualizations** - Interactive crop growth & data analytics

### 💬 Intelligent Chatbot

- **RAG System** - Retrieval-Augmented Generation for contextual answers
- **Local Language Support** - Answers in local languages
- **Knowledge Base** - 48+ crops, 100+ cultivation guides
- **Smart Response Generation** - LLM-enhanced replies

### 🔒 Security & Compliance

- ✅ JWT Authentication
- ✅ Role-based Access Control (RBAC)
- ✅ Encrypted Data Storage
- ✅ HTTPS/TLS Support
- ✅ Rate Limiting & DDoS Protection

---

## Tech Stack

### Backend
```
FastAPI 0.123+          | Modern, fast Python web framework
SQLAlchemy 2.0+        | Database ORM & queries
Pydantic 2.0+          | Data validation & serialization
PyTorch 2.0+           | Deep learning framework
Transformers           | State-of-the-art NLP & Vision models
scikit-learn           | Classical ML algorithms
Paho-MQTT              | IoT message broker integration
```

### Frontend
```
React 18.3+            | UI library
TypeScript 5+          | Type-safe JavaScript
Vite                   | Lightning-fast build tool
TailwindCSS 3+         | Utility-first styling
React Query            | Data fetching & caching
shadcn/ui              | Accessible component library
Three.js               | 3D visualizations
```

### Infrastructure
```
SQLite                 | Development database
PostgreSQL             | Production database
Docker & Docker Compose| Containerization
Azure Container Apps   | Cloud deployment
Azure Cosmos DB        | Production NoSQL database
```

### IoT Platforms
```
ESP32 Microcontroller  | Main sensor hub
Arduino Nano           | Temperature module
PlatformIO             | Firmware development
Arduino IDE            | Sketch uploading
```

---

## Quick Start

### Prerequisites
- Python 3.12.10+
- Node.js 20.x LTS+
- Git
- (Optional) Docker Desktop

### 1️⃣ Clone Repository

```bash
git clone https://github.com/ELANGKATHIR11/AgriSense-A-Smart-Agriculture-Solution.git
cd AGRISENSEFULL-STACK
```

### 2️⃣ Backend Setup

```bash
cd src/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from core.data_store import init_sensor_db; init_sensor_db()"

# Start server (runs on http://localhost:8000)
uvicorn main:app --reload
```

### 3️⃣ Frontend Setup

```bash
cd src/frontend

# Install dependencies
npm install

# Start development server (runs on http://localhost:5173)
npm run dev
```

### 4️⃣ Access Application

- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **Admin Dashboard**: http://localhost:5173/admin

---

## Project Structure

```
AGRISENSEFULL-STACK/
├── src/
│   ├── backend/          # FastAPI REST API
│   │   ├── api/          # Route handlers
│   │   ├── ml/           # ML model services
│   │   ├── iot/          # IoT data ingestion
│   │   ├── models/       # Database models
│   │   └── main.py       # Entry point
│   │
│   └── frontend/         # React web interface
│       ├── src/
│       │   ├── components/
│       │   ├── pages/
│       │   └── lib/
│       └── package.json
│
├── iot-devices/         # Microcontroller firmware
│   └── AGRISENSE_IoT/
│       ├── esp32_firmware/
│       └── arduino_nano_firmware/
│
├── tests/               # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e-tests/
│
├── documentation/       # Comprehensive docs
│   ├── api/            # API documentation
│   ├── guides-docs/    # User guides
│   ├── architecture-docs/
│   ├── ml-models/
│   └── security/
│
├── guides/             # Quick references
├── deployment/         # Docker configs
└── scripts/            # Utility scripts
```

**See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed directory information.**

---

## Documentation

### 📚 Main Guides

| Document | Purpose |
|----------|---------|
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Complete project organization |
| [ARCHITECTURE_DIAGRAM.md](guides/ARCHITECTURE_DIAGRAM.md) | System design & diagrams |
| [documentation/README.md](documentation/README.md) | Documentation index |

### 🔧 Setup Guides

| Guide | Topic |
|-------|-------|
| [API Documentation](documentation/api/API_DOCUMENTATION.md) | REST API endpoints |
| [IoT Firmware](iot-devices/AGRISENSE_IoT/) | Sensor configuration |
| [Database Schema](documentation/architecture-docs/) | Data models |

### 🤖 AI/ML Documentation

| Document | Focus |
|----------|-------|
| [ML Models](documentation/ml-models/) | Model architecture & performance |
| [Chatbot Reference](guides/CHATBOT_QUICK_REFERENCE.md) | Chatbot features |
| [Evaluation Report](guides/ML_MODEL_EVALUATION_COMPREHENSIVE_REPORT.md) | Model metrics |

### 🔐 Security & Best Practices

| Document | Topic |
|----------|-------|
| [Security Hardening](documentation/security/SECURITY_HARDENING.md) | Security guidelines |
| [Copilot Instructions](.github/copilot-instructions.md) | Development standards |

---

## 🚀 Deployment

### Local Docker Deployment

```bash
cd deployment/docker
docker-compose up -d

# Access at http://localhost:5173
```

### Azure Deployment

```bash
# Create Azure resources
az group create --name agrisense-rg --location eastus

# Deploy container app
az containerapp up --name agrisense-backend \
  --resource-group agrisense-rg \
  --source ./src/backend \
  --ingress external \
  --target-port 8000
```

See [guides/ARCHITECTURE_DIAGRAM.md](guides/ARCHITECTURE_DIAGRAM.md) for detailed cloud setup.

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# E2E tests
npm run test:e2e

# All tests with coverage
pytest --cov=src/backend tests/ --cov-report=html
```

---

## 📊 17 ML Models Included

| # | Model | Type | Purpose |
|---|-------|------|---------|
| 1 | crop_recommendation_rf | Random Forest | Crop selection |
| 2 | crop_recommendation_gb | Gradient Boosting | Crop ranking |
| 3 | crop_recommendation_nn | Neural Network | Smart recommendations |
| 4-5 | crop_recommendation_tf | TensorFlow DNN | Deep learning classification |
| 6 | yield_prediction | Regression | Yield forecasting |
| 7 | water_model | Random Forest | Irrigation optimization |
| 8 | fertilizer_model | Regressor | NPK recommendations |
| 9 | disease_model | CNN Transfer Learning | Disease detection |
| 10 | weed_model | Segmentation | Weed identification |
| 11 | intent_classifier | SVM/LogReg | Chatbot routing |
| 12 | chatbot_encoder | SBERT | Semantic search |
| 13-14 | optimized_models | Ensemble | Enhanced predictions |
| 15-16 | enhanced_detection | Joblib | Fine-tuned models |
| 17 | openvino_npu | Intel NPU | Accelerated inference |

---

## 🎓 Learning Resources

- **FastAPI**: https://fastapi.tiangolo.com/
- **React**: https://react.dev/
- **Machine Learning**: https://scikit-learn.org/
- **IoT with ESP32**: https://docs.espressif.com/

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for coding standards.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**ELANGKATHIR11**
- GitHub: [@ELANGKATHIR11](https://github.com/ELANGKATHIR11)
- Repository: [AgriSense](https://github.com/ELANGKATHIR11/AgriSense-A-Smart-Agriculture-Solution)

---

## 🙏 Acknowledgments

- **TensorFlow & PyTorch** - Deep learning frameworks
- **FastAPI Team** - Modern web framework
- **React Community** - UI library
- **Hugging Face** - Pre-trained models
- **OpenWeather** - Weather API integration

---

## 📞 Support & Issues

- **Issues**: [GitHub Issues](https://github.com/ELANGKATHIR11/AgriSense-A-Smart-Agriculture-Solution/issues)
- **Documentation**: [/documentation](/documentation/README.md)
- **Quick Reference**: [/guides](/guides/)

---

<div align="center">

**🌾 AgriSense - Empowering Farmers with Technology 🚀**

*Making sustainable agriculture accessible to everyone*

[![GitHub Stars](https://img.shields.io/github/stars/ELANGKATHIR11/AgriSense-A-Smart-Agriculture-Solution?style=social)](https://github.com/ELANGKATHIR11/AgriSense-A-Smart-Agriculture-Solution)
[![GitHub Forks](https://img.shields.io/github/forks/ELANGKATHIR11/AgriSense-A-Smart-Agriculture-Solution?style=social)](https://github.com/ELANGKATHIR11/AgriSense-A-Smart-Agriculture-Solution)

</div>
