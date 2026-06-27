# 🌾 AgriSense - AI-Powered Agricultural Platform

**Full-stack AI/ML application for smart farming with real-time monitoring, crop disease detection, and intelligent recommendations.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-blue.svg)](https://reactjs.org/)

---

## 🚀 Features

### 🔬 AI/ML Capabilities
- **Crop Disease Detection** - Computer vision with TensorFlow & PyTorch
- **Intelligent Recommendations** - ML-powered crop advisory
- **Chatbot Assistant** - NLP-based farming guidance
- **Predictive Analytics** - Time-series forecasting for yields

### 📊 Real-Time Monitoring
- **IoT Sensor Integration** - ESP32/Arduino compatibility
- **Live Dashboard** - React-based monitoring interface
- **Automated Alerts** - SMS/Email notifications
- **Historical Analytics** - Trend analysis and insights

### 🏗️ Architecture
- **Backend:** FastAPI (Python 3.12) with async/await
- **Frontend:** React 18 with TypeScript + Vite
- **ML Stack:** TensorFlow 2.18, PyTorch 2.5, Transformers
- **Background Tasks:** Celery with Redis broker
- **Database:** MongoDB (production) / SQLite (development)

---

## 🌐 Live Demo

**Access the application:**
```
https://huggingface.co/spaces/<your-username>/agrisense-app
```

**API Documentation (Swagger):**
```
https://huggingface.co/spaces/<your-username>/agrisense-app/docs
```

---

## 📖 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- MongoDB (Atlas M0 free tier)
- Redis (Upstash free tier)

### Local Development

```bash
# Clone repository
git clone https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK.git
cd AGRISENSEFULL-STACK

# Backend setup
cd agrisense_app/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend/farm-fortune-frontend-main
npm install
npm run build

# Start services
# Terminal 1: Backend
cd agrisense_app/backend
uvicorn main:app --reload --port 8004

# Terminal 2: Celery
celery -A celery_config worker --loglevel=info

# Terminal 3: Frontend (dev mode)
cd agrisense_app/frontend/farm-fortune-frontend-main
npm run dev
```

---

## 🐳 Docker Deployment

### Build & Run Locally

```bash
# Build Docker image
docker build -f Dockerfile.huggingface -t agrisense .

# Run container
docker run -p 7860:7860 \
  -e MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/agrisense" \
  -e REDIS_URL="redis://default:pass@host:6379" \
  -e AGRISENSE_ADMIN_TOKEN="your-secret-token" \
  agrisense
```

### Deploy to Hugging Face Spaces

See complete guide: [HF_DEPLOYMENT_GUIDE.md](HF_DEPLOYMENT_GUIDE.md)

**Quick Steps:**
1. Create Space with Docker SDK
2. Add secrets (`MONGO_URI`, `REDIS_URL`, `AGRISENSE_ADMIN_TOKEN`)
3. Push code to Space repository
4. Wait for build (~10-15 minutes)

---

## 🔐 Environment Variables

### Required
| Variable | Description | Example |
|----------|-------------|---------|
| `MONGO_URI` | MongoDB connection string | `mongodb+srv://user:pass@cluster.mongodb.net/agrisense` |
| `REDIS_URL` | Redis connection URL | `redis://default:pass@host:6379` |
| `AGRISENSE_ADMIN_TOKEN` | Admin API token | `sk-agrisense-xyz123` |

### Optional
| Variable | Default | Description |
|----------|---------|-------------|
| `AGRISENSE_DISABLE_ML` | `0` | Set to `1` to disable ML models |
| `WORKERS` | `2` | Number of Uvicorn workers |
| `CELERY_WORKERS` | `2` | Number of Celery workers |
| `LOG_LEVEL` | `info` | Logging level |

---

## 📊 API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /docs` - API documentation

### Sensors
- `POST /api/sensors/readings` - Submit sensor data
- `GET /api/sensors/readings` - Fetch sensor history

### AI/ML
- `POST /api/predict/disease` - Crop disease detection
- `POST /api/recommendations` - Get farming recommendations
- `POST /api/chat` - Chatbot interaction

### User Management
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login

---

## 🧪 Testing

```bash
# Backend tests
cd agrisense_app/backend
pytest tests/ -v --cov

# Frontend tests
cd agrisense_app/frontend/farm-fortune-frontend-main
npm run test

# E2E tests
npm run test:e2e
```

---

## 📈 Performance

### Resource Usage (16GB RAM Container)
- **Idle:** ~2GB RAM
- **With ML Models Loaded:** ~6-8GB RAM
- **Under Load:** ~10-12GB RAM
- **CPU:** 2-4 cores typical usage

### Optimization Tips
1. Enable model lazy loading
2. Use TensorFlow Lite for inference
3. Implement Redis caching
4. Optimize database queries

---

## 🛠️ Tech Stack

### Backend
- **Framework:** FastAPI 0.115+
- **Language:** Python 3.12.10
- **ML/AI:** TensorFlow 2.18, PyTorch 2.5, Transformers 4.47
- **Task Queue:** Celery 5.4 with Redis
- **Database:** MongoDB (Motor), SQLite (SQLAlchemy)
- **Authentication:** JWT with FastAPI-Users

### Frontend
- **Framework:** React 18.3
- **Language:** TypeScript 5.3
- **Build Tool:** Vite 5.0
- **UI Components:** Radix UI, Tailwind CSS
- **State Management:** TanStack Query
- **Testing:** Vitest, Playwright

### Infrastructure
- **Compute:** Hugging Face Spaces (Docker SDK)
- **Database:** MongoDB Atlas M0 (free tier)
- **Cache/Broker:** Upstash Redis (free tier)
- **Monitoring:** Prometheus, Sentry (optional)

---

## 📁 Project Structure

```
AGRISENSEFULL-STACK/
├── agrisense_app/
│   ├── backend/
│   │   ├── main.py                 # FastAPI application
│   │   ├── celery_config.py        # Celery configuration
│   │   ├── requirements.txt        # Python dependencies
│   │   ├── api/                    # API routes
│   │   ├── core/                   # Business logic
│   │   ├── ml/                     # ML model loaders
│   │   └── tasks/                  # Celery tasks
│   └── frontend/
│       └── farm-fortune-frontend-main/
│           ├── src/                # React source code
│           ├── package.json        # Node dependencies
│           └── vite.config.ts      # Vite configuration
├── ml_models/                      # Trained ML models
├── Dockerfile.huggingface          # Multi-stage Dockerfile
├── start.sh                        # Container startup script
└── HF_DEPLOYMENT_GUIDE.md         # Deployment guide
```

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** - Free 16GB RAM hosting
- **MongoDB Atlas** - Free M0 tier database
- **Upstash** - Free Redis hosting
- **FastAPI** - Modern Python web framework
- **React** - Frontend library

---

## 📧 Contact

**Project Maintainer:** ELANGKATHIR11

- GitHub: [@ELANGKATHIR11](https://github.com/ELANGKATHIR11)
- Repository: [AGRISENSEFULL-STACK](https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK)

---

## 🔗 Links

- **Documentation:** [Full Documentation](HF_DEPLOYMENT_GUIDE.md)
- **API Docs:** [Swagger UI](https://your-space.hf.space/docs)
- **GitHub:** [Source Code](https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK)
- **Hugging Face:** [Live Demo](https://huggingface.co/spaces/<username>/agrisense-app)

---

**Built with ❤️ for sustainable agriculture**
