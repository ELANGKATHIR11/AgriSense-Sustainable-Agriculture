# 🌾 AgriSense — Sustainable Agriculture IoT Platform

AgriSense is a full-stack agricultural IoT platform that combines real-time sensor monitoring, machine-learning-driven crop recommendations, and an optional offline-capable Edge AI service to help farmers make data-driven decisions.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Service & Port Reference](#service--port-reference)
- [Quick Start (Recommended Path)](#quick-start-recommended-path)
- [Environment Variable Setup](#environment-variable-setup)
- [Advanced & Optional Paths](#advanced--optional-paths)
- [Runbook](#runbook)
- [Documentation Map](#documentation-map)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      Browser / Client                   │
│            React + Vite + TypeScript  (:3001)           │
└────────────────────────┬────────────────────────────────┘
                         │ REST / WebSocket
┌────────────────────────▼────────────────────────────────┐
│              Node.js / Express API  (:5000)             │
│   Auth · IoT ingestion · Route aggregation · Swagger    │
└──────┬────────────────────────────┬─────────────────────┘
       │ REST                       │ REST
┌──────▼──────────────┐   ┌─────────▼──────────────────┐
│  Python ML Service  │   │  Edge AI Service (optional) │
│  FastAPI  (:8000)   │   │  Flask  (:5002)             │
│  Crop/Yield/Water   │   │  Offline chatbot + vision   │
│  recommendations    │   │  (no internet required)     │
└─────────────────────┘   └────────────────────────────┘
              │
       ┌──────▼──────┐
       │  SQLite /   │
       │  MongoDB /  │
       │  PostgreSQL │
       └─────────────┘
```

IoT devices (ESP32 / Arduino) push sensor readings to the Node.js API over HTTP or MQTT.

---

## Service & Port Reference

| Service | Technology | Default Port | Start command |
|---|---|---|---|
| Frontend | React + Vite | **3001** | `npm run dev:frontend` |
| Backend API | Node.js / Express | **5000** | `npm run dev:backend` |
| ML Service | Python / FastAPI | **8000** | `cd backend/ml && uvicorn fastapi_service:app --reload` |
| Edge AI (optional) | Python / Flask | **5002** | `cd backend/ml && python edge_ai_service.py` |

---

## Quick Start (Recommended Path)

### Prerequisites

| Tool | Minimum version |
|---|---|
| Node.js | 18 LTS |
| npm | 9 |
| Python | 3.9 |

### 1 — Clone and install dependencies

```bash
git clone <repository-url>
cd AgriSense-Sustainable-Agriculture

# Install all Node dependencies (root + backend + frontend)
npm run install:all
```

### 2 — Configure environment variables

```bash
# Copy the templates and fill in your values
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

Edit each `.env` file with your local settings (database URLs, API keys, etc.).
See [Environment Variable Setup](#environment-variable-setup) for details.

### 3 — Set up the Python ML service

```bash
cd backend/ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit as needed
```

### 4 — Start all services

**Option A — single terminal (recommended for development):**

```bash
npm run dev:all
```

This starts the Node backend and React frontend concurrently.
Start the Python ML service in a separate terminal:

```bash
cd backend/ml && source .venv/bin/activate && uvicorn fastapi_service:app --port 8000 --reload
```

**Option B — separate terminals:**

```bash
# Terminal 1 — Node backend
npm run dev:backend

# Terminal 2 — React frontend
npm run dev:frontend

# Terminal 3 — Python ML service
cd backend/ml && uvicorn fastapi_service:app --port 8000 --reload
```

### 5 — Verify everything is running

| URL | Expected response |
|---|---|
| http://localhost:5000/api/health | `{"message":"OK", ...}` |
| http://localhost:5000/api/health/ready | `{"ready":true, ...}` |
| http://localhost:3001 | Frontend dashboard |
| http://localhost:8000/docs | FastAPI Swagger UI |

---

## Environment Variable Setup

### Root `.env` (orchestration)

Copy `.env.example` → `.env` at the repository root.  
Used when running both services together via npm scripts.

### Backend `.env`

Copy `backend/.env.example` → `backend/.env`.  
Key variables:

| Variable | Description | Default |
|---|---|---|
| `PORT` | Node.js server port | `5000` |
| `NODE_ENV` | Runtime environment | `development` |
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017/agrisense` |
| `JWT_SECRET` | Secret for signing JWTs | *(set a strong random value)* |
| `ML_SERVICE_URL` | URL of the Python ML service | `http://localhost:8000` |
| `FRONTEND_URL` | CORS allowed origin | `http://localhost:3001` |
| `USE_EDGE_AI` | Enable Edge AI service | `false` |

### Frontend `.env`

Copy `frontend/.env.example` → `frontend/.env`.

| Variable | Description | Default |
|---|---|---|
| `VITE_API_BASE_URL` | Backend API base URL | `http://localhost:5000` |
| `VITE_ML_SERVICE_URL` | ML service base URL | `http://localhost:8000` |
| `VITE_APP_TITLE` | Browser tab title | `AgriSense` |

### Python ML service `.env`

Copy `backend/ml/.env.example` → `backend/ml/.env`.

| Variable | Description | Default |
|---|---|---|
| `ML_PORT` | FastAPI service port | `8000` |
| `ML_WORKERS` | Uvicorn worker count | `1` |
| `USE_GPU` | Enable GPU acceleration | `false` |
| `MODELS_DIR` | Path to trained model files | `./models` |

> **Security note:** Never commit `.env` files. All secrets must stay local or in a secrets manager.

---

## Advanced & Optional Paths

### Edge AI (offline) service

For offline environments without internet access, the Edge AI service provides a local chatbot and vision inference:

```bash
cd backend/ml
python edge_ai_service.py         # Flask service on :5002
python edge_ai_chatbot.py         # Local NLP chatbot
```

Set `USE_EDGE_AI=true` in `backend/.env` to route API calls through the edge service.

See [`backend/ml/README_EDGE_AI.md`](backend/ml/README_EDGE_AI.md) for setup details.

### IoT / Sensor integration

Firmware and hardware setup for ESP32 and Arduino Nano sensors:

```
AGRISENSE_IoT/
├── esp32_firmware/     # ESP32 firmware + PlatformIO config
└── arduino_nano_firmware/  # Arduino bridge scripts
```

See [`AGRISENSE_IoT/arduino_nano_firmware/README.md`](AGRISENSE_IoT/arduino_nano_firmware/README.md).

### VLM / Training pipeline

Advanced vision-language model training utilities live in `backend/ml/`. See:
- [`backend/ml/VLM_SETUP_STATUS.md`](backend/ml/VLM_SETUP_STATUS.md)
- [`backend/ml/VLM_TRAINING_COMPLETE.md`](backend/ml/VLM_TRAINING_COMPLETE.md)

### Production deployment

See [`SETUP_CHECKLIST.md`](SETUP_CHECKLIST.md) and [`RUN.md`](RUN.md) for production configuration guidance.

---

## Runbook

### Startup order

Start services in this order to avoid dependency failures:

1. **Database** (MongoDB / PostgreSQL) — must be running before Node API
2. **Python ML service** (port 8000) — start before or alongside Node API
3. **Node.js backend** (port 5000)
4. **React frontend** (port 3001)

### Healthcheck endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Basic liveness check |
| `GET /api/health/live` | Kubernetes liveness probe |
| `GET /api/health/ready` | Kubernetes readiness probe (DB + ML service) |
| `GET /api/health/detailed` | Full status including all dependencies |
| `GET /docs` | FastAPI auto-generated API docs (ML service) |

### Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ECONNREFUSED :5000` | Node backend not started | Run `npm run dev:backend` |
| `ECONNREFUSED :8000` | Python ML service not started | Start FastAPI service (see above) |
| `MongoServerError: connect ECONNREFUSED` | MongoDB not running | Start MongoDB: `mongod` |
| Frontend blank page / 404 | Vite dev server not running | Run `npm run dev:frontend` |
| `JWT malformed` | Missing or wrong `JWT_SECRET` | Check `backend/.env` |
| ML predictions return 503 | `USE_ML_SERVICE=false` or ML service down | Set `USE_ML_SERVICE=true`, start ML service |
| CORS error in browser | `FRONTEND_URL` mismatch | Set `FRONTEND_URL=http://localhost:3001` in `backend/.env` |
| Python `ModuleNotFoundError` | venv not activated or deps not installed | `source backend/ml/.venv/bin/activate && pip install -r requirements.txt` |

---

## Documentation Map

| Document | Contents |
|---|---|
| `README.md` *(this file)* | Project overview, quick-start, runbook |
| [`SETUP_CHECKLIST.md`](SETUP_CHECKLIST.md) | Step-by-step environment setup checklist |
| [`RUN.md`](RUN.md) | Detailed run instructions for all services |
| [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) | Common commands cheat-sheet |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Detailed troubleshooting guide |
| [`CLAUDE_INTEGRATION_GUIDE.md`](CLAUDE_INTEGRATION_GUIDE.md) | AI assistant integration details |
| [`INTEGRATION_SUMMARY.md`](INTEGRATION_SUMMARY.md) | Service integration architecture summary |
| [`AGRISENSE_BLUEPRINT.md`](AGRISENSE_BLUEPRINT.md) | Full system blueprint and design decisions |
| [`PROJECT_BLUEPRINT.md`](PROJECT_BLUEPRINT.md) | Project roadmap and milestones |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Extended operational runbook |
| [`backend/ml/README_EDGE_AI.md`](backend/ml/README_EDGE_AI.md) | Edge AI service setup |
| [`.github/CI_SECRETS_GUIDE.md`](.github/CI_SECRETS_GUIDE.md) | GitHub Actions secrets configuration |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Follow the quick-start above to set up locally
4. Make your changes and run `npm test` (backend) and frontend tests
5. Open a pull request — CI will validate your changes automatically

---

## License

[MIT](LICENSE)
