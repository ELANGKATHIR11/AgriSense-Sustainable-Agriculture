# AgriSense Project Structure

## 📁 Directory Organization

```
AGRISENSEFULL-STACK/
├── src/                          # Core application source code
│   ├── backend/                  # FastAPI backend (Python)
│   │   ├── api/                 # API route handlers
│   │   ├── ai/                  # AI/ML services
│   │   ├── auth/                # Authentication & security
│   │   ├── core/                # Core utilities & config
│   │   ├── iot/                 # IoT data ingestion
│   │   ├── ml/                  # Machine learning models
│   │   ├── middleware/          # Express middleware
│   │   ├── models/              # Database models & schemas
│   │   ├── routes/              # API endpoint definitions
│   │   ├── nlp/                 # NLP & chatbot logic
│   │   ├── vlm/                 # Vision Language Model services
│   │   ├── integrations/        # External service integrations
│   │   ├── utils/               # Helper functions
│   │   ├── websocket_manager.py # WebSocket handling
│   │   ├── main.py              # Application entry point
│   │   └── requirements.txt     # Python dependencies
│   │
│   └── frontend/                 # React + Vite frontend
│       ├── src/                 # Source files
│       │   ├── components/      # React components
│       │   ├── pages/           # Page components
│       │   ├── lib/             # Utilities & API client
│       │   ├── hooks/           # Custom React hooks
│       │   └── App.tsx          # Main App component
│       ├── public/              # Static assets
│       ├── package.json         # Node dependencies
│       └── vite.config.ts       # Vite configuration
│
├── iot-devices/                  # IoT firmware & configurations
│   └── AGRISENSE_IoT/
│       ├── esp32_firmware/      # ESP32 sensor firmware
│       ├── arduino_nano_firmware/ # Arduino temperature sensor
│       └── esp32_config.py      # IoT configuration
│
├── deployment/                   # Docker & deployment files
│   └── docker/                  # Docker configurations
│       ├── Dockerfile
│       └── docker-compose.yml
│
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── e2e-tests/               # End-to-end tests
│   ├── conftest.py              # Test configuration
│   └── fixtures.py              # Test fixtures
│
├── scripts/                      # Utility scripts
│   ├── deploy.sh                # Deployment scripts
│   └── ...                      # Other utilities
│
├── documentation/               # Comprehensive documentation
│   ├── README.md               # Documentation index
│   ├── api/                    # API documentation
│   ├── guides-docs/            # User & developer guides
│   ├── architecture-docs/      # Architecture documentation
│   ├── ml-models/              # ML model documentation
│   ├── security/               # Security guidelines
│   └── images/                 # Documentation images
│
├── guides/                      # Quick reference guides
│   ├── ARCHITECTURE_DIAGRAM.md
│   ├── DOCUMENTATION_INDEX.md
│   ├── PROJECT_ORGANIZATION.md
│   ├── CHATBOT_QUICK_REFERENCE.md
│   └── ML_MODEL_EVALUATION_COMPREHENSIVE_REPORT.md
│
├── .github/                     # GitHub configuration
│   └── copilot-instructions.md # Copilot guidelines
│
├── .env.example                # Environment variables template
├── .env.production.template    # Production env template
├── .gitignore                  # Git ignore rules
├── package.json               # Root dependencies (if any)
├── tsconfig.json              # TypeScript configuration
├── pytest.ini                 # Pytest configuration
├── playwright.config.ts       # E2E test configuration
├── README.md                  # Main project README
├── ARCHITECTURE_DIAGRAM.md    # System architecture
└── openapi.json               # OpenAPI specification
```

## 📚 Key Directories

### `/src/backend/`
- **Purpose**: FastAPI REST API server
- **Key Files**: `main.py`, `requirements.txt`
- **Subdirs**: `api/`, `ml/`, `iot/`, `models/`, `auth/`
- **Run**: `uvicorn main:app --reload`

### `/src/frontend/`
- **Purpose**: React + Vite web interface
- **Key Files**: `package.json`, `vite.config.ts`
- **Subdirs**: `src/components/`, `src/pages/`, `public/`
- **Run**: `npm run dev`

### `/iot-devices/`
- **Purpose**: Microcontroller firmware for sensors
- **ESP32**: WiFi-enabled sensor hub (DHT22, pH, moisture, etc.)
- **Arduino**: Temperature sensor module
- **Configuration**: `esp32_config.py`

### `/tests/`
- **Unit Tests**: Business logic validation
- **Integration Tests**: API endpoint testing
- **E2E Tests**: Full workflow testing
- **Run**: `pytest` or `npm run test:e2e`

### `/documentation/`
- **API Docs**: OpenAPI/Swagger specifications
- **Guides**: Step-by-step guides for features
- **Architecture**: System design & diagrams
- **Security**: Best practices & compliance

### `/guides/`
- Quick reference documentation
- Project organization overview
- Chatbot features reference
- ML model evaluation results

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `.env.example` | Environment variables template |
| `.env.production.template` | Production settings template |
| `pytest.ini` | Pytest test runner config |
| `tsconfig.json` | TypeScript compiler options |
| `playwright.config.ts` | E2E test framework config |
| `openapi.json` | REST API specification |

## 🚀 Quick Start

### Backend Setup
```bash
cd src/backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup
```bash
cd src/frontend
npm install
npm run dev
```

### IoT Firmware
```bash
# ESP32
cd iot-devices/AGRISENSE_IoT/esp32_firmware
# Use PlatformIO or Arduino IDE to flash

# Arduino Nano
cd iot-devices/AGRISENSE_IoT/arduino_nano_firmware
# Use Arduino IDE to flash
```

## 📖 Documentation Index

- **Main README**: Root `README.md` - Project overview
- **Architecture**: `guides/ARCHITECTURE_DIAGRAM.md` - System design
- **API Reference**: `documentation/api/API_DOCUMENTATION.md`
- **Developer Guide**: `documentation/guides-docs/` - Development guides
- **ML Models**: `documentation/ml-models/` - Model documentation
- **Security**: `documentation/security/SECURITY_HARDENING.md`

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

## 📦 Dependencies Management

### Backend Dependencies
- **Core**: FastAPI, SQLAlchemy, Pydantic
- **ML/AI**: PyTorch, Transformers, scikit-learn
- **IoT**: paho-mqtt
- **See**: `src/backend/requirements.txt`

### Frontend Dependencies
- **Framework**: React 18.3+, TypeScript
- **Build**: Vite, TailwindCSS
- **State**: React Query, Zustand
- **UI**: shadcn/ui components
- **See**: `src/frontend/package.json`

## 🔐 Environment Variables

Create `.env` file from `.env.example`:

```env
# Database
DATABASE_URL=sqlite:///./sensors.db

# API Settings
DEBUG=true
LOG_LEVEL=INFO

# LLM Integration (Optional)
PHI_LLM_ENDPOINT=http://localhost:11434
PHI_MODEL_NAME=phi:latest

# External APIs
OPENWEATHER_API_KEY=your_key
OPENAI_API_KEY=your_key
```

## 📝 File Cleanup Summary

The following have been removed for a clean structure:
- ✅ 60+ cleanup/report scripts
- ✅ 40+ temporary documentation files
- ✅ Duplicate training scripts
- ✅ Redundant module files
- ✅ Temporary analysis reports
- ✅ Outdated guide files

This results in a **lean, well-organized codebase** ready for development and deployment.

## 🔄 Deployment

### Local Development
```bash
# Terminal 1: Backend
cd src/backend && uvicorn main:app --reload

# Terminal 2: Frontend
cd src/frontend && npm run dev
```

### Docker Deployment
```bash
cd deployment/docker
docker-compose up -d
```

### Azure Deployment
See `guides/ARCHITECTURE_DIAGRAM.md` for cloud setup instructions.

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: `/documentation/` folder
- **Quick Reference**: `/guides/` folder
- **Architecture**: `guides/ARCHITECTURE_DIAGRAM.md`

---

**Last Updated**: January 2026
**Status**: Production Ready ✅
