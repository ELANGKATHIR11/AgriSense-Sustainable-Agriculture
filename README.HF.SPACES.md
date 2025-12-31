---
title: AgriSense
emoji: 🌾
colorFrom: green
colorTo: green
sdk: docker
pinned: true
---

# 🌾 AgriSense - AI-Powered Smart Farming Platform

**AgriSense** is a comprehensive full-stack agricultural IoT platform combining real-time sensor monitoring, AI-powered crop disease detection, weed management, and intelligent farming recommendations.

## ✨ Features

### 📊 Real-Time Monitoring
- **IoT Sensor Integration**: ESP32 & Arduino-based environmental sensors
- **Live Dashboard**: Real-time temperature, humidity, soil moisture, pH tracking
- **Data Visualization**: Interactive charts and field maps
- **Weather Integration**: Current weather and crop-specific recommendations

### 🤖 AI/ML Capabilities
- **Disease Detection**: CNN-powered plant disease identification
- **Weed Management**: Computer vision for weed identification and control
- **Yield Prediction**: ML models for crop yield forecasting
- **Smart Farming**: Crop-specific optimization recommendations
- **Conversational AI**: AgriBot chatbot for farming advice

### 💾 Data Management
- **MongoDB Atlas** backend for scalable data storage
- **Redis caching** for high-performance queries
- **Celery workers** for asynchronous background tasks
- **RESTful API** with FastAPI

### 🎨 Modern Frontend
- **React 18** with TypeScript
- **Real-time updates** via WebSocket
- **Responsive Design** for desktop and mobile
- **Multi-language support** (English, Hindi, Tamil, Telugu, Kannada)
- **Progressive Web App** capabilities

## 🚀 Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI 0.115+ (Python 3.12.10) |
| **Frontend** | React 18.3 + Vite 5.0 + TypeScript |
| **Database** | MongoDB Atlas M0 (FREE) |
| **Cache** | Upstash Redis (FREE) |
| **Job Queue** | Celery 5.4 |
| **ML/AI** | TensorFlow 2.18, PyTorch 2.5, Transformers 4.47 |
| **Compute** | Hugging Face Spaces Docker (16GB RAM) |

## 📱 Live API Documentation

Once deployed:
- **API Docs**: `/docs` (Swagger UI)
- **Alternative Docs**: `/redoc` (ReDoc)
- **Health Check**: `/health`

## 🔌 API Endpoints

### Sensor Data
- `POST /api/v1/sensors/readings` - Submit sensor readings
- `GET /api/v1/sensors/{device_id}/latest` - Get latest readings
- `GET /api/v1/sensors/{device_id}/history` - Historical data

### AI/ML Models
- `POST /api/v1/ai/disease-detection` - Detect plant diseases
- `POST /api/v1/ai/weed-detection` - Identify weeds
- `POST /api/v1/ai/recommendations` - Get farming recommendations
- `POST /api/v1/ai/chat` - Talk to AgriBot chatbot

### Analytics
- `GET /api/v1/analytics/yield-prediction` - Predict yield
- `GET /api/v1/analytics/crop-health` - Health metrics
- `GET /api/v1/analytics/field-summary` - Field statistics

## 🛠️ Deployment Configuration

This Space uses:
- **Docker SDK**: Full Python + Node.js environment
- **Environment Variables**: MongoDB, Redis URLs (set in Secrets)
- **Multi-process**: FastAPI + Celery workers + Nginx

## 📊 Free Tier Stack

| Service | Tier | Cost |
|---------|------|------|
| Hugging Face Spaces | CPU Basic (16GB) | FREE |
| MongoDB Atlas | M0 Sandbox (512MB) | FREE |
| Upstash Redis | Free tier | FREE |
| **TOTAL MONTHLY COST** | | **$0** ✅ |

## 🌐 Access Points

- **Frontend UI**: `/ui/` (React app)
- **API Server**: FastAPI on port 8000
- **API Docs**: `/docs`
- **Health**: `/health`

## 📈 Monitoring & Logs

Monitor the Space build and runtime:
- Check logs in the Space settings
- Monitor API health at `/health`
- View real-time metrics in dashboard

## 🔧 Configuration

The application uses these environment secrets:
- `MONGO_URI` - MongoDB Atlas connection string
- `REDIS_URL` - Upstash Redis URL
- `AGRISENSE_ADMIN_TOKEN` - Admin authentication token

## 📚 Documentation

- [API Documentation](./docs/)
- [Installation Guide](./INSTALLATION.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Architecture Overview](./ARCHITECTURE.md)

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows PEP 8 (Python) and Prettier (TypeScript)
- All tests pass before submitting PR
- Documentation is updated

## 📝 License

MIT License - See LICENSE file for details

## 🙋 Support & Contact

For issues, feature requests, or questions:
- Open an issue on GitHub
- Contact: agrisense@example.com

---

**Made with 💚 for sustainable agriculture**

*AgriSense helps farmers make data-driven decisions to increase crop yields, reduce waste, and optimize resource usage.*
