# AgriSense - Smart Farming Intelligence Platform

<div align="center">
  <img src="agrisense_app/frontend/farm-fortune-frontend-main/src/assets/branding/agrisense_logo.svg" alt="AgriSense Logo" width="420" />
  
  <p><em>Transforming Agriculture with AI, IoT, and Precision Farming</em></p>
  
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.123+-teal.svg)](https://fastapi.tiangolo.com)
  [![React](https://img.shields.io/badge/React-18+-cyan.svg)](https://reactjs.org)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue.svg)](https://typescriptlang.org)
  [![Security](https://img.shields.io/badge/vulnerabilities-0-brightgreen.svg)](PYTHON_312_UPGRADE_SUMMARY.md)
</div>

> **🎉 Latest Update (December 2025)**: Fully upgraded to **Python 3.12.10** with all dependencies at latest versions. See [PYTHON_312_UPGRADE_SUMMARY.md](PYTHON_312_UPGRADE_SUMMARY.md) for details.

## 🌱 About AgriSense

AgriSense is a comprehensive smart agriculture platform that combines cutting-edge AI, IoT sensors, and precision farming techniques to optimize crop yields, reduce water consumption, and promote sustainable farming practices.

### Key Features

🧠 **AI-Powered Analytics**
- Machine learning models for crop recommendations
- Predictive analytics for irrigation scheduling
- Disease and weed detection algorithms
- Intelligent chatbot for farming guidance

📡 **IoT Integration**
- Real-time sensor monitoring
- Automated irrigation control
- Tank level and valve management
- Weather data integration

🌾 **Precision Farming**
- Soil analysis and recommendations
- Crop-specific nutrient planning
- Water usage optimization
- Yield prediction and planning

💧 **Water Management**
- Smart irrigation scheduling
- Rainwater harvesting tracking
- Drought-resistant farming techniques
- Water conservation metrics

## ⚡ NPU Optimization (NEW!)

**Intel Core Ultra 9 275HX NPU Acceleration** - Train models 2-10x faster, run inference 10-50x faster!

```powershell
# Quick setup for NPU-optimized training
.\setup_npu_environment.ps1
.\venv_npu\Scripts\Activate.ps1
python tools/npu/train_npu_optimized.py
```

📖 **Documentation**: [NPU_OPTIMIZATION_GUIDE.md](NPU_OPTIMIZATION_GUIDE.md) | [NPU_QUICK_START.md](NPU_QUICK_START.md)

**Benefits**:
- 🚀 10-50x faster inference on NPU
- ⚡ 2-10x faster training with Intel oneDAL
- 💾 4x smaller models (INT8 quantization)
- ⚡ 5x lower power consumption

---

## 🚀 Quick Start

### Local Development

Run the complete application with a single command:

```powershell
.\start-agrisense.ps1
```

**Access Points:**
- **Main Application**: http://localhost:8004/ui
- **API Documentation**: http://localhost:8004/docs
- **Admin Panel**: http://localhost:8004/ui/admin

### 🌐 Deploy to Hugging Face Spaces (FREE - 16GB RAM)

**Complete deployment in 30-45 minutes with zero cost:**

```bash
# 1. One-command automated setup
bash deploy_to_huggingface.sh agrisense-app your-username

# 2. Add secrets in Space settings (MONGO_URI, REDIS_URL, AGRISENSE_ADMIN_TOKEN)
# 3. Push and wait for build (~10-15 minutes)
# 4. Access at: https://huggingface.co/spaces/your-username/agrisense-app
```

**What's Included (100% FREE):**
- 16GB RAM compute (Hugging Face Spaces)
- MongoDB Atlas M0 sandbox (512MB)
- Upstash Redis free tier (10K commands/day)
- Multi-stage Docker container
- FastAPI + Celery orchestration
- Frontend static serve

📖 **See** [HF_DEPLOYMENT_GUIDE.md](HF_DEPLOYMENT_GUIDE.md) for complete setup instructions.

## 📱 Application Features

### Core Modules

| Module | Description | Key Features |
|--------|-------------|--------------|
| **Dashboard** | Central monitoring hub | Real-time metrics, system status, alerts |
| **Crops Database** | Comprehensive crop catalog | 2000+ crops, search, detailed profiles |
| **AI Chatbot** | Intelligent farming assistant | Natural language Q&A, context-aware help |
| **Soil Analysis** | Soil health assessment | pH, nutrients, moisture analysis |
| **Irrigation Control** | Smart water management | Automated scheduling, remote control |
| **Disease Management** | Plant health monitoring | AI-powered disease detection |
| **Weed Management** | Weed identification & control | Computer vision classification |
| **Live Monitoring** | Real-time sensor data | IoT dashboard, alerts, trends |
 
---

## ⚙️ Runtime system libraries (for ML / OpenCV)

If you plan to run the ML/CV-enabled image or use OpenCV locally, the runtime environment needs a few OS-level graphics libraries so OpenCV and related packages can load `libGL.so.1`. The optimized Dockerfile includes these, but for local troubleshooting or CI runners install the following on Debian/Ubuntu hosts:

```bash
sudo apt-get update && sudo apt-get install -y \
  libgl1-mesa-glx libgl1-mesa-dri libglib2.0-0 libsm6 libxrender1 libxext6
```

If you see errors mentioning `libGL.so.1` at runtime, installing the packages above in the image or host will typically resolve the issue.


### Smart Features

🔬 **ML Models** (18 trained models, ~400MB)
- Water recommendation AI (291.87MB)
- Fertilizer optimization AI (83.37MB)
- Crop yield prediction
- Disease classification
- Weed identification

📊 **Data Analytics**
- Historical trend analysis
- Predictive modeling
- Cost-benefit analysis
- Environmental impact tracking

## 🛠 Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLite** - Lightweight database for sensors
- **TensorFlow** - Machine learning models
- **scikit-learn** - Classical ML algorithms
- **MQTT** - IoT device communication

### Frontend
- **React 18** - Modern UI framework
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Vite** - Fast build tooling
- **Lucide Icons** - Beautiful icon system

### Infrastructure
- **Docker** - Containerized deployment
- **Azure** - Cloud hosting platform
- **GitHub Actions** - CI/CD automation
- **PWA** - Progressive web app capabilities

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- Git

### Development Setup

1. **Clone Repository**
```bash
git clone https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK.git
cd AGRISENSEFULL-STACK
```

2. **Backend Setup**
```powershell
cd "AGRISENSEFULL-STACK\agrisense_app\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. **Frontend Setup**
```powershell
cd "agrisense_app\frontend\farm-fortune-frontend-main"
npm install
```

4. **Start Application**
```powershell
# Use the integrated startup script
.\start-agrisense.ps1

# Or start manually:
# Backend: uvicorn agrisense_app.backend.main:app --port 8004
# Frontend: npm run dev (development) or served via backend (production)
```

### Environment Variables

```bash
# Optional ML acceleration (default: enabled)
AGRISENSE_DISABLE_ML=1

# API configuration
AGRISENSE_ADMIN_TOKEN=your_admin_token

# Database paths (auto-configured)
DB_PATH=./sensors.db
```

## 🌍 Deployment

### Production Deployment
```bash
# Build frontend assets
cd agrisense_app/frontend/farm-fortune-frontend-main
npm run build

# Start production server
cd ../../backend
uvicorn main:app --host 0.0.0.0 --port 8004
```

### Docker Deployment
```bash
docker build -t agrisense .
docker run -p 8004:8004 agrisense
```

### Azure Cloud Deployment
```bash
# Deploy using Azure CLI
az webapp up --name agrisense-app --resource-group agriculture-rg
```

## 📚 Documentation

- [API Documentation](http://localhost:8004/docs) - Interactive Swagger UI
- [Project Blueprint](docs/PROJECT_BLUEPRINT.md) - Comprehensive technical overview
- [UI/UX Guide](UI_UX_ENHANCEMENT_REPORT.md) - Design system documentation
- [ML Model Inventory](ML_MODEL_INVENTORY.md) - AI/ML architecture details
- [Branding Guide](agrisense_app/frontend/farm-fortune-frontend-main/src/assets/branding/BRANDING_GUIDE.md) - Logo usage guidelines

## 🧪 Testing & Quality

### Automated Testing
```powershell
# Run comprehensive test suite
.\comprehensive_test_suite.ps1

# Individual test categories
pytest scripts/test_backend_inprocess.py  # Backend API tests
pytest scripts/test_edge_endpoints.py     # Edge device tests
npm test                                  # Frontend unit tests
```

### Performance Metrics
- **Backend Response**: <50ms average API response time
- **Frontend Loading**: <2s initial page load
- **ML Inference**: <100ms per recommendation
- **Database Queries**: <10ms for sensor data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow TypeScript/Python type hints
- Maintain test coverage >80%
- Use conventional commit messages
- Update documentation for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Agricultural Research Organizations** - Domain expertise and validation
- **Open Source Community** - Libraries and frameworks
- **Farming Communities** - Real-world testing and feedback
- **Environmental Scientists** - Sustainability guidance

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK/discussions)
- **Email**: [Contact Team](mailto:support@agrisense.com)

---

<div align="center">
  <img src="agrisense_app/frontend/farm-fortune-frontend-main/src/assets/branding/agrisense_logo_icon.svg" alt="AgriSense Icon" width="64" />
  
  **Built with ❤️ for sustainable agriculture and the future of farming**
  
  [Website](https://agrisense.com) • [Documentation](docs/) • [Demo](http://localhost:8004/ui)
</div>