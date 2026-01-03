# AgriSense Project Organization Guide

**Last Updated**: January 3, 2026  
**Organization Status**: ✅ COMPLETE  

---

## 📋 Table of Contents

1. [Directory Structure Overview](#directory-structure-overview)
2. [Quick Navigation Guide](#quick-navigation-guide)
3. [File Organization Details](#file-organization-details)
4. [How to Find Things](#how-to-find-things)
5. [Adding New Files](#adding-new-files)
6. [Best Practices](#best-practices)

---

## 📁 Directory Structure Overview

```
AgriSense/
│
├── 📂 src/                          # SOURCE CODE (All application code)
│   ├── backend/                     # Backend API & Services
│   │   ├── ai/                      # AI/ML services & chatbots
│   │   ├── auth/                    # Authentication & authorization
│   │   ├── core/                    # Core functionality
│   │   ├── database/                # Database layer
│   │   ├── integrations/            # External service integrations
│   │   ├── middleware/              # FastAPI middleware
│   │   ├── models/                  # SQLAlchemy/Pydantic models
│   │   ├── nlp/                     # NLP services
│   │   ├── notifications/           # Notification services
│   │   ├── monitoring/              # Metrics & monitoring
│   │   ├── routes/                  # API endpoints
│   │   ├── security/                # Security utilities
│   │   ├── agrisense_app/           # Main app directory
│   │   ├── main.py                  # FastAPI entry point
│   │   ├── requirements.txt         # Python dependencies
│   │   └── requirements-ml.txt      # Optional ML dependencies
│   │
│   ├── frontend/                    # React/TypeScript Frontend
│   │   ├── farm-fortune-frontend/   # Main frontend app
│   │   ├── components/              # React components
│   │   ├── pages/                   # Page components
│   │   ├── assets/                  # Images, icons, styles
│   │   ├── hooks/                   # Custom React hooks
│   │   ├── lib/                     # Utility libraries
│   │   ├── package.json             # Node.js dependencies
│   │   └── tsconfig.json            # TypeScript config
│   │
│   └── iot/                         # IoT Device Code
│       ├── arduino/                 # Arduino sketches (.ino)
│       ├── esp32/                   # ESP32 firmware
│       └── edge/                    # Edge computing code
│
├── 📂 data/                         # DATA & DATASETS
│   ├── datasets/                    # Training & reference data
│   │   ├── Crop_recommendation.csv
│   │   ├── india_crop_dataset.csv
│   │   └── *.csv                    # Other datasets
│   │
│   └── training-data/               # Training-specific data
│       ├── synthetic_data/
│       └── preprocessed/
│
├── 📂 models/                       # ML MODELS
│   ├── pretrained/                  # Pre-trained models
│   │   ├── *.pb                     # TensorFlow models
│   │   ├── *.bin                    # ONNX/PyTorch models
│   │   └── README.md                # Model documentation
│   │
│   ├── trained/                     # Trained models
│   │   ├── *.joblib                 # Scikit-learn models
│   │   ├── *.pkl                    # Pickle models
│   │   ├── *.h5                     # Keras/TensorFlow models
│   │   ├── *.pt                     # PyTorch models
│   │   ├── *.onnx                   # ONNX models
│   │   └── README.md                # Training details
│   │
│   └── documentation/               # Model documentation
│       ├── crop_models.md
│       ├── disease_detection.md
│       ├── weed_detection.md
│       └── yield_prediction.md
│
├── 📂 tests/                        # TEST SUITE
│   ├── unit/                        # Unit tests
│   │   ├── test_backend.py
│   │   ├── test_models.py
│   │   ├── test_api.py
│   │   └── ...
│   │
│   ├── integration/                 # Integration tests
│   │   ├── test_db_integration.py
│   │   ├── test_api_integration.py
│   │   └── ...
│   │
│   ├── e2e/                         # End-to-end tests
│   │   ├── playwright/              # Playwright tests
│   │   ├── test_workflows.py
│   │   └── ...
│   │
│   ├── performance/                 # Performance tests
│   │   ├── locustfile.py            # Load testing
│   │   ├── benchmark.py
│   │   └── ...
│   │
│   ├── conftest.py                  # Pytest configuration
│   ├── pytest.ini                   # Pytest settings
│   └── README.md                    # Testing guide
│
├── 📂 docs/                         # DOCUMENTATION
│   ├── guides/                      # User guides & tutorials
│   │   ├── QUICKSTART.md            # Getting started
│   │   ├── CHATBOT_GUIDE.md
│   │   ├── ML_MODELS_GUIDE.md
│   │   ├── GENAI_INTEGRATION.md
│   │   ├── README.md                # All guides index
│   │   └── ...
│   │
│   ├── api/                         # API documentation
│   │   ├── api_reference.md
│   │   ├── endpoints.md
│   │   └── openapi.json
│   │
│   ├── setup/                       # Setup & installation
│   │   ├── DEVELOPMENT.md           # Dev environment setup
│   │   ├── DEPLOYMENT.md            # Production deployment
│   │   ├── CUDA_SETUP.md
│   │   ├── NPU_SETUP.md
│   │   ├── WSL2_SETUP.md
│   │   └── ENV_VARS.md
│   │
│   ├── architecture/                # Architecture docs
│   │   ├── ARCHITECTURE.md          # System design
│   │   ├── DESIGN.md
│   │   └── diagrams/
│   │
│   ├── deployment/                  # Deployment docs
│   │   ├── DOCKER.md
│   │   ├── KUBERNETES.md
│   │   ├── AZURE.md
│   │   └── HF_SPACES.md
│   │
│   ├── troubleshooting/             # Troubleshooting guides
│   │   ├── COMMON_ISSUES.md
│   │   ├── FAQ.md
│   │   └── DEBUG.md
│   │
│   ├── api-reference/               # API reference
│   │   ├── openapi.json
│   │   └── schema.md
│   │
│   └── archived/                    # Old/obsolete docs
│       └── ...
│
├── 📂 scripts/                      # AUTOMATION SCRIPTS
│   ├── deployment/                  # Deployment scripts
│   │   ├── start.py                 # Main startup script
│   │   ├── start_agrisense.ps1
│   │   ├── start_backend_gpu.sh
│   │   └── README.md
│   │
│   ├── setup/                       # Environment setup
│   │   ├── install_cuda_wsl2.ps1
│   │   ├── setup_npu_environment.ps1
│   │   ├── setup_environment.py
│   │   └── README.md
│   │
│   ├── training/                    # Model training
│   │   ├── train.py                 # Main training script
│   │   ├── retrain_gpu.py
│   │   ├── retrain_production.py
│   │   ├── train_npu_models.ps1
│   │   └── README.md
│   │
│   ├── monitoring/                  # Monitoring scripts
│   │   ├── monitor_training.ps1
│   │   ├── monitor_api.py
│   │   └── README.md
│   │
│   └── utilities/                   # Utility scripts
│       ├── validate_*.py
│       ├── cleanup_*.py
│       ├── check_*.py
│       └── README.md
│
├── 📂 config/                       # CONFIGURATION
│   ├── environments/                # Environment files
│   │   ├── .env.example
│   │   ├── .env.production.template
│   │   ├── .env.development
│   │   └── .env.test
│   │
│   ├── docker/                      # Docker configs
│   │   ├── Dockerfile
│   │   ├── Dockerfile.ml
│   │   ├── Dockerfile.optimized
│   │   ├── docker-compose.yml
│   │   └── .dockerignore
│   │
│   ├── .gitignore                   # Git ignore rules
│   ├── tsconfig.json                # TypeScript config
│   ├── playwright.config.ts         # E2E test config
│   └── README.md                    # Configuration guide
│
├── 📂 reports/                      # PROJECT REPORTS
│   ├── analysis/                    # Analysis reports
│   │   ├── E2E_ANALYSIS_REPORT.json
│   │   ├── analysis_report.json
│   │   └── README.md
│   │
│   ├── cleanup/                     # Cleanup records
│   │   ├── E2E_CLEANUP_REPORT.md
│   │   ├── E2E_CLEANUP_PLAN.md
│   │   ├── CLEANUP_LOG_*.json
│   │   └── README.md
│   │
│   ├── performance/                 # Performance metrics
│   │   ├── ML_MODEL_TEST_RESULTS.json
│   │   ├── retraining_report_*.json
│   │   └── README.md
│   │
│   └── benchmarks/                  # Benchmark results
│       ├── npu_benchmark_results.json
│       ├── gpu_benchmarks.json
│       └── README.md
│
├── 📂 tools/                        # DEVELOPMENT TOOLS
│   ├── development/                 # Dev tools
│   │   ├── code_generator.py
│   │   ├── blueprint_generator.py
│   │   ├── comprehensive_analysis.py
│   │   └── README.md
│   │
│   ├── optimization/                # Optimization tools
│   │   ├── performance_optimizer.py
│   │   ├── model_optimizer.py
│   │   └── README.md
│   │
│   ├── security/                    # Security tools
│   │   ├── security_audit.py
│   │   ├── dependency_checker.py
│   │   └── README.md
│   │
│   └── analysis/                    # Analysis tools
│       ├── code_analyzer.py
│       ├── dependency_analyzer.py
│       └── README.md
│
├── 📂 examples/                     # EXAMPLE CODE
│   ├── api-usage/                   # API usage examples
│   │   ├── crop_prediction.py
│   │   ├── disease_detection.py
│   │   └── chatbot_usage.py
│   │
│   ├── ml-models/                   # ML model examples
│   │   ├── train_crop_model.py
│   │   ├── inference_example.py
│   │   └── fine_tuning.py
│   │
│   └── integration/                 # Integration examples
│       ├── hybrid_ai.py
│       ├── vlm_integration.py
│       └── rag_example.py
│
├── 📂 notebooks/                    # JUPYTER NOTEBOOKS
│   ├── eda/                         # Exploratory data analysis
│   ├── model_training/              # Model training notebooks
│   ├── analysis/                    # Data analysis notebooks
│   └── README.md
│
├── 📄 README.md                     # Main project README
├── 📄 ORGANIZATION_GUIDE.md         # This file
├── 📄 LICENSE
├── 📄 package.json                  # Root package.json
├── 📄 requirements-npu.txt          # NPU requirements
├── 📄 .github/                      # GitHub workflows & config
├── 📄 .vscode/                      # VS Code settings
└── 📄 .env.example                  # Example environment
```

---

## 🧭 Quick Navigation Guide

### Finding Source Code
- **Backend**: `src/backend/`
- **Frontend**: `src/frontend/`
- **IoT Code**: `src/iot/`
- **AI/ML Services**: `src/backend/ai/`
- **API Routes**: `src/backend/routes/`
- **Database Models**: `src/backend/models/`

### Finding Data
- **Training Datasets**: `data/datasets/`
- **Training Data**: `data/training-data/`
- **Model Weights**: `models/trained/` and `models/pretrained/`

### Finding Tests
- **Unit Tests**: `tests/unit/`
- **Integration Tests**: `tests/integration/`
- **End-to-End Tests**: `tests/e2e/`
- **Performance Tests**: `tests/performance/`

### Finding Documentation
- **Setup Guides**: `docs/setup/`
- **API Docs**: `docs/api/` or `docs/api-reference/`
- **Deployment**: `docs/deployment/`
- **Architecture**: `docs/architecture/`
- **Troubleshooting**: `docs/troubleshooting/`
- **General Guides**: `docs/guides/`

### Finding Scripts
- **Start Application**: `scripts/deployment/start.py`
- **Train Models**: `scripts/training/train.py`
- **Setup Environment**: `scripts/setup/`
- **Monitor System**: `scripts/monitoring/`
- **Utilities**: `scripts/utilities/`

### Finding Configuration
- **Environment Variables**: `config/environments/`
- **Docker Setup**: `config/docker/`
- **App Config**: `config/`

### Finding Reports
- **Analysis**: `reports/analysis/`
- **Cleanup Records**: `reports/cleanup/`
- **Performance**: `reports/performance/`
- **Benchmarks**: `reports/benchmarks/`

---

## 📊 File Organization Details

### Backend Code Organization

```
src/backend/ai/                     # All AI/ML services
├── chatbot_conversational.py        # Conversational chatbot
├── chatbot_phi_integration.py       # Phi model integration
├── disease_detection.py             # Disease detection model
├── crop_classification.py           # Crop classification
├── weed_management.py               # Weed detection & management
├── yield_prediction.py              # Yield prediction model
├── nlp_services.py                  # NLP utilities
├── ml_features.py                   # ML feature extraction
├── rag_adapter.py                   # RAG adapter for retrieval
├── vlm_engine.py                    # Vision-language models
├── plant_health_monitor.py          # Plant health monitoring
├── smart_farming_ml.py              # Smart farming algorithms
└── README.md                        # AI services guide

src/backend/integrations/           # External integrations
├── llm_clients.py                   # LLM API clients (OpenAI)
├── weather.py                       # Weather API integration
├── vlm_scold_integration.py         # SCOLD VLM integration
└── README.md                        # Integration guide

src/backend/database/               # Database layer
├── database_enhanced.py             # Enhanced DB operations
├── models.py                        # SQLAlchemy models
└── README.md                        # Database guide

src/backend/auth/                   # Authentication
├── auth.py                          # Main auth logic
├── auth_enhanced.py                 # Enhanced auth
├── security.py                      # Security utilities
└── README.md                        # Auth guide
```

### Test Organization

```
tests/unit/                         # Unit tests
├── test_backend.py                  # Backend tests
├── test_api_endpoints.py            # API tests
├── test_models.py                   # Model tests
├── test_auth.py                     # Auth tests
└── test_ml.py                       # ML service tests

tests/integration/                  # Integration tests
├── test_api_integration.py          # Full API integration
├── test_db_integration.py           # Database integration
├── test_ml_pipeline.py              # ML pipeline tests
└── test_external_services.py        # External service tests

tests/e2e/                          # End-to-end tests
├── test_user_workflows.py           # User workflow tests
├── test_api_flows.py                # API flow tests
├── e2e_local_runner.py              # Local E2E runner
└── playwright/                      # Playwright tests
    ├── test_frontend.spec.ts
    └── test_workflows.spec.ts

tests/performance/                  # Performance tests
├── locustfile.py                    # Load testing
├── benchmark.py                     # Performance benchmark
└── stress_test.py                   # Stress tests
```

### Documentation Organization

```
docs/guides/                        # General guides
├── QUICKSTART.md                    # Get started quickly
├── CHATBOT_GUIDE.md                 # Using the chatbot
├── ML_MODELS_GUIDE.md               # Using ML models
├── GENAI_INTEGRATION.md             # GenAI features
├── ARCHITECTURE_DIAGRAM.md          # System architecture
├── PROJECT_STRUCTURE.md             # Project structure
├── README.md                        # Guides index
└── ... more guides

docs/setup/                         # Setup documentation
├── DEVELOPMENT.md                   # Dev environment setup
├── DEPLOYMENT.md                    # Deployment guide
├── CUDA_SETUP.md                    # CUDA installation
├── NPU_SETUP.md                     # NPU optimization
├── WSL2_SETUP.md                    # WSL2 setup
├── ENV_VARS.md                      # Environment variables
└── README.md                        # Setup guide index

docs/api/                           # API documentation
├── api_reference.md                 # Full API reference
├── endpoints.md                     # Endpoint documentation
└── README.md

docs/deployment/                    # Deployment docs
├── DOCKER.md                        # Docker deployment
├── KUBERNETES.md                    # K8s deployment
├── AZURE.md                         # Azure deployment
├── HF_SPACES.md                     # Hugging Face Spaces
└── README.md
```

---

## 🔍 How to Find Things

### Find Backend Files
```
Location: src/backend/
Examples:
  - API routes: src/backend/routes/
  - Models: src/backend/models/
  - AI services: src/backend/ai/
  - Integrations: src/backend/integrations/
```

### Find Frontend Files
```
Location: src/frontend/farm-fortune-frontend/
Examples:
  - Components: src/frontend/farm-fortune-frontend/src/components/
  - Pages: src/frontend/farm-fortune-frontend/src/pages/
  - Styles: src/frontend/farm-fortune-frontend/src/assets/
```

### Find Test Files
```
Location: tests/
Categories:
  - Unit tests: tests/unit/
  - Integration tests: tests/integration/
  - E2E tests: tests/e2e/
  - Performance tests: tests/performance/
```

### Find ML Models
```
Location: models/
Categories:
  - Pre-trained: models/pretrained/
  - Trained: models/trained/
  - Documentation: models/documentation/
```

### Find Configuration
```
Location: config/
Examples:
  - Environment files: config/environments/.env.*
  - Docker configs: config/docker/
  - Application config: config/
```

### Find Documentation
```
Location: docs/
Categories:
  - Setup: docs/setup/
  - Guides: docs/guides/
  - API: docs/api/
  - Deployment: docs/deployment/
  - Architecture: docs/architecture/
```

### Find Scripts
```
Location: scripts/
Categories:
  - Deployment: scripts/deployment/start.py
  - Training: scripts/training/train.py
  - Setup: scripts/setup/
  - Monitoring: scripts/monitoring/
  - Utilities: scripts/utilities/
```

---

## ➕ Adding New Files

### Backend Code
Add to `src/backend/` in appropriate subdirectory:
```
- AI/ML code → src/backend/ai/
- Authentication → src/backend/auth/
- Database → src/backend/database/
- Routes → src/backend/routes/
- Models → src/backend/models/
- Integrations → src/backend/integrations/
```

### Frontend Code
Add to `src/frontend/farm-fortune-frontend/`:
```
- Components → src/frontend/farm-fortune-frontend/src/components/
- Pages → src/frontend/farm-fortune-frontend/src/pages/
- Hooks → src/frontend/farm-fortune-frontend/src/hooks/
- Assets → src/frontend/farm-fortune-frontend/src/assets/
```

### Tests
Add to `tests/` in appropriate subdirectory:
```
- Unit tests → tests/unit/
- Integration tests → tests/integration/
- E2E tests → tests/e2e/
- Performance tests → tests/performance/
```

### Documentation
Add to `docs/` in appropriate subdirectory:
```
- Setup guides → docs/setup/
- User guides → docs/guides/
- API docs → docs/api/
- Deployment docs → docs/deployment/
- Architecture docs → docs/architecture/
```

### Models
Add to `models/` in appropriate subdirectory:
```
- Pre-trained models → models/pretrained/
- Trained models → models/trained/
- Model documentation → models/documentation/
```

### Scripts
Add to `scripts/` in appropriate subdirectory:
```
- Deployment scripts → scripts/deployment/
- Setup scripts → scripts/setup/
- Training scripts → scripts/training/
- Monitoring scripts → scripts/monitoring/
- Utility scripts → scripts/utilities/
```

---

## ✅ Best Practices

### 1. File Naming Conventions
```
Python Files:
  - use_snake_case.py
  - test_module_name.py
  - conftest.py (for pytest fixtures)

Documentation:
  - UPPERCASE_NAMES.md for main docs
  - lowercase_for_specific.md for detailed docs

Configuration:
  - .env.example for templates
  - .env.production for production

Directories:
  - use-lowercase-with-hyphens/ for new dirs
  - descriptive names (e.g., src, tests, docs)
```

### 2. Code Organization
- Keep related code together
- Use meaningful module names
- Add README.md in major directories
- Document public APIs

### 3. Documentation
- Add docstrings to all functions/classes
- Keep guides up to date
- Update ORGANIZATION_GUIDE.md for new structure
- Link related documentation

### 4. Directory Rules
- One responsibility per directory
- Don't mix different types of files
- Keep __init__.py in Python packages
- Add README.md explaining directory purpose

### 5. Moving Files
When reorganizing:
1. Update import paths in code
2. Update documentation references
3. Update CI/CD pipeline if needed
4. Create migration guide if breaking change

---

## 📞 Quick Reference

| What | Where |
|------|-------|
| Start Application | `scripts/deployment/start.py` |
| Train Models | `scripts/training/train.py` |
| Run Tests | `tests/` or `pytest` |
| API Documentation | `docs/api/` or `/docs` endpoint |
| Environment Setup | `scripts/setup/` or `docs/setup/` |
| Model Files | `models/pretrained/` or `models/trained/` |
| Datasets | `data/datasets/` |
| Configuration | `config/` |

---

## 🎯 Next Steps

1. ✅ All files have been organized
2. ⬜ Update import paths in code that reference old locations
3. ⬜ Update CI/CD pipelines if needed
4. ⬜ Create comprehensive README for each major directory
5. ⬜ Add navigation links to all README files

---

**Version**: 1.0  
**Last Updated**: January 3, 2026  
**Status**: ✅ COMPLETE

