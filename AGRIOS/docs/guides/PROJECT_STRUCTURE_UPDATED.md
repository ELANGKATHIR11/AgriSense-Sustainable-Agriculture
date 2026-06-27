# AgriSense Project Structure (Updated Jan 3, 2026)

## 📁 Core Application Structure

```
AgriSense/
├── agrisense_app/                    # Main application
│   ├── backend/                      # FastAPI backend
│   │   ├── main.py                   # Application entry point
│   │   ├── requirements.txt          # Core Python dependencies (cleaned up)
│   │   ├── requirements-ml.txt       # Optional ML dependencies
│   │   ├── requirements-dev.txt      # Development dependencies
│   │   │
│   │   ├── api/                      # API routes
│   │   │   ├── routes/               # Endpoint handlers
│   │   │   ├── models.py             # Pydantic request/response models
│   │   │   └── dependencies.py       # Dependency injection
│   │   │
│   │   ├── core/                     # Core functionality
│   │   │   ├── config.py             # Configuration management
│   │   │   ├── security.py           # Authentication & security
│   │   │   └── middleware.py         # FastAPI middleware
│   │   │
│   │   ├── ai/                       # AI & ML services
│   │   │   ├── ml/                   # Machine learning models
│   │   │   ├── chatbot_*.py          # Chatbot implementations
│   │   │   ├── disease_detection.py  # Disease detection
│   │   │   └── crop_*.py             # Crop-related models
│   │   │
│   │   ├── integrations/             # External service integrations
│   │   │   ├── llm_clients.py        # LLM integration (OpenAI)
│   │   │   ├── weather.py            # Weather API integration
│   │   │   └── vlm_*.py              # Vision-language model integration
│   │   │
│   │   ├── models/                   # Database models
│   │   │   └── *.py                  # SQLAlchemy/Pydantic models
│   │   │
│   │   ├── database/                 # Database layer (optional)
│   │   │   └── database_*.py         # DB configuration
│   │   │
│   │   ├── tasks/                    # Background tasks
│   │   │   └── celery_config.py      # Celery configuration
│   │   │
│   │   └── sensors.db                # SQLite database (development)
│   │
│   ├── frontend/                     # React/TypeScript frontend
│   │   └── farm-fortune-frontend-main/
│   │       ├── src/
│   │       │   ├── components/       # React components
│   │       │   ├── pages/            # Page components
│   │       │   ├── lib/              # Utility libraries
│   │       │   ├── hooks/            # Custom React hooks
│   │       │   └── assets/           # Images, styles, etc.
│   │       ├── package.json          # Node.js dependencies (E2E tests)
│   │       └── tsconfig.json         # TypeScript configuration
│   │
│   └── scripts/                      # Utility scripts
│       ├── start.py                  # Unified startup script (replaces 7 variants)
│       ├── train.py                  # Unified training script (replaces 8 variants)
│       └── deploy.sh                 # Deployment helper
│
├── tests/                            # Test suite
│   ├── conftest.py                   # Pytest configuration
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── e2e/                          # End-to-end tests
│
├── AGRISENSE_IoT/                    # IoT device code
│   ├── ESP32/                        # ESP32 firmware
│   └── Arduino/                      # Arduino sketches
│
├── agrisense_pi_edge_minimal/        # Raspberry Pi edge deployment
│
├── AI_Models/                        # Pre-trained models
│   ├── crop_*.joblib                 # Crop classification models
│   ├── disease_*.joblib              # Disease detection models
│   ├── weed_*.joblib                 # Weed detection models
│   └── *.pb                          # TensorFlow models
│
├── datasets/                         # Training data
│   ├── Crop_recommendation.csv
│   ├── india_crop_dataset.csv
│   └── synthetic_train.py
│
├── docker/                           # Docker configurations
│   ├── Dockerfile.optimized          # Optimized production image
│   ├── Dockerfile.ml                 # ML-enabled image
│   └── docker-compose.yml            # Compose configuration
│
├── config/                           # Configuration files
│   ├── development.yaml              # Development config
│   └── production.yaml               # Production config
│
├── documentation/                    # Project documentation
│   ├── ARCHITECTURE_DIAGRAM.md       # System architecture
│   ├── DEPLOYMENT.md                 # Deployment guide (CONSOLIDATED)
│   ├── DEVELOPMENT.md                # Development guide (NEW)
│   ├── QUICKSTART.md                 # Quick start guide (NEW)
│   ├── TESTING.md                    # Testing procedures (NEW)
│   ├── ENV_VARS_REFERENCE.md         # Environment variables
│   ├── E2E_TESTING_GUIDE.md          # E2E testing
│   ├── HARDWARE_OPTIMIZATION_CONFIG.md
│   ├── PYTHON_312_QUICK_REFERENCE.md # Python 3.12 reference
│   ├── CUDA_QUICK_START.md           # CUDA setup
│   ├── NPU_QUICK_START.md            # NPU optimization
│   ├── WSL2_CUDA_SETUP_GUIDE.md      # WSL2 setup
│   └── archived/                     # Old/obsolete docs (MOVED)
│       ├── *_REPORT.md
│       ├── *_SUMMARY.md
│       ├── *_EVALUATION.md
│       └── ... (150+ obsolete files)
│
├── tools/                            # Development tools
│   ├── development/                  # Development utilities
│   ├── npu/                          # NPU optimization tools
│   ├── security_audit.py             # Security scanning
│   └── generate_blueprint.py         # Code generation
│
├── examples/                         # Example code
│   └── *.py                          # Usage examples
│
├── e2e/                              # E2E test configuration
│   ├── playwright.config.ts
│   └── tests/
│
├── scripts/                          # Standalone scripts
│   ├── setup_repo.ps1                # Repository setup
│   ├── validate_*.py                 # Validation utilities
│   └── ... other utilities
│
├── .github/
│   ├── workflows/                    # CI/CD workflows
│   ├── copilot-instructions.md       # GitHub Copilot config
│   └── pull_request_template.md
│
├── .vscode/
│   └── settings.json                 # VS Code settings
│
├── .env.example                      # Example environment variables
├── .env.production.template          # Production template
├── .gitignore                        # Git ignore rules (UPDATED)
├── pytest.ini                        # Pytest configuration
├── tsconfig.json                     # TypeScript configuration
├── playwright.config.ts              # Playwright E2E config
│
├── README.md                         # Main README (UPDATED)
├── DOCUMENTATION_INDEX.md            # Documentation index
└── ... other project files

```

---

## 🗂️ Cleaned Up (Deleted)

### Virtual Environments (4.0 GB Recovered)
- ❌ `venv312/` - 1.2 GB
- ❌ `venv_ml312/` - 2.7 GB
- ❌ `venv_npu/` - 1.1 GB (partial - locked files)
- ❌ `.venv/`, `.venv312/`, `.venv.bak/`

### Duplicate Script Files
- ❌ `start_agrisense.bat` → Consolidated
- ❌ `start.sh` → Consolidated  
- ❌ `start_agrisense_scold.ps1` → Consolidated
- ❌ `start_hybrid_ai.ps1` → Consolidated
- ⚠️ `retrain_*.py` variants - kept for backward compatibility

### Obsolete Reports (38 files removed)
- ❌ `CLEANUP_*.md` - All cleanup reports
- ❌ `OPTIMIZATION_*.md` - Optimization reports
- ❌ `*_SUMMARY.md` - Duplicate summaries
- ❌ `*_REPORT.md` - Old evaluation reports
- ❌ `GPU_TRAINING_SESSION_SUMMARY.md`
- ❌ `NPU_TRAINING_SESSION_SUMMARY.md`
- ❌ And 30+ others

### Temporary Files
- ❌ `tmp_*.py` - Temporary test files
- ❌ `*.log` - Log files
- ❌ `temp_*.onnx.data` - Temporary models
- ❌ `.file_sizes.json`, `.sizes_summary.json`, `.pip_freeze.txt`

### Backup Directories
- ❌ `cleanup_backup_20251205_182237/`

---

## 📊 Space Recovery Summary

| Category | Size | Status |
|----------|------|--------|
| Virtual Environments | 4.0 GB | ✅ Deleted |
| Reports & Documentation | 0.3 MB | ✅ Deleted |
| Temporary Files | 80+ MB | ✅ Deleted |
| Duplicate Scripts | 30 KB | ✅ Cleaned |
| **Total Recovered** | **~4.0 GB** | **✅ COMPLETE** |

---

## 🚀 Updated Startup Instructions

### Single Entry Point (UNIFIED)
```powershell
# All-in-one startup script (replaces 7 variants)
python scripts/start.py --help
python scripts/start.py --backend --frontend
python scripts/start.py --all  # Start everything
```

### Unified Training
```powershell
# Single training script (replaces 8 variants)
python scripts/train.py --help
python scripts/train.py --gpu
python scripts/train.py --npu --fast
python scripts/train.py --production
```

---

## 📚 Documentation Status

### Actively Maintained ✅
- `README.md` - Updated
- `ARCHITECTURE_DIAGRAM.md` - Current
- `DOCUMENTATION_INDEX.md` - Updated
- `ENV_VARS_REFERENCE.md` - Current
- `E2E_TESTING_GUIDE.md` - Current

### Recently Created 🆕
- `DEPLOYMENT.md` - Consolidated deployment guide
- `DEVELOPMENT.md` - Development setup
- `QUICKSTART.md` - Quick start for new devs
- `TESTING.md` - Testing procedures

### Reference/Specialized 📖
- `CUDA_QUICK_START.md`
- `NPU_QUICK_START.md`  
- `WSL2_CUDA_SETUP_GUIDE.md`
- `HARDWARE_OPTIMIZATION_CONFIG.md`
- `PYTHON_312_QUICK_REFERENCE.md`

### Archived 📦
- All `*_REPORT.md` files → `/documentation/archived/`
- All `*_SUMMARY.md` files → `/documentation/archived/`
- All evaluation & optimization reports → `/documentation/archived/`

---

## 🔧 Configuration Files (Preserved)

- `.gitignore` - Updated with venv patterns
- `.env.example` - Environment template
- `.env.production.template` - Production template
- `pytest.ini` - Test configuration
- `tsconfig.json` - TypeScript config
- `playwright.config.ts` - E2E config

---

## 🎯 Key Changes (Jan 3, 2026 Cleanup)

1. **Removed 4 GB of virtual environments** - Never commit these!
2. **Consolidated duplicate scripts** - Single entry point for startup/training
3. **Removed 38 obsolete reports** - Cleaned project clutter
4. **Updated documentation** - New guides for DEPLOYMENT, DEVELOPMENT, QUICKSTART, TESTING
5. **Updated `.gitignore`** - Prevents future venv commits
6. **Generated E2E analysis** - Documented all changes in E2E_CLEANUP_PLAN.md

---

## 🚀 Next Steps for Development

1. Create new virtual environment: `python -m venv venv312`
2. Activate environment: `venv312/Scripts/activate`
3. Install dependencies: `pip install -r agrisense_app/backend/requirements.txt`
4. Follow [QUICKSTART.md](documentation/QUICKSTART.md) for setup

