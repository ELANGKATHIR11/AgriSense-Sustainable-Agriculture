# AgriSense E2E Cleanup & Documentation Update Plan

**Generated**: January 3, 2026  
**Project Size**: 522.07 MB  
**Recoverable Space**: ~5.2 GB (from virtual environments alone)

## 📊 Analysis Summary

### File Inventory
- **Total Files**: ~2,000+ files
- **Python Scripts**: 324 files
- **Documentation**: 143 markdown files
- **Test Files**: Multiple duplicates (test*.py, test*.ps1, test*.sh)
- **Configuration**: 108 JSON files
- **Model Files**: 29 joblib, 8 .pb, 4 .h5, 4 .pkl, 4 .onnx

### 🚨 Critical Issues Found

#### 1. **Virtual Environments (5.1 GB)**
- `venv312`: 1200.34 MB
- `venv_ml312`: 2726.68 MB
- `venv_npu`: 1144.0 MB
- **Status**: Should be in .gitignore only, not committed
- **Action**: DELETE immediately

#### 2. **Duplicate Startup Scripts** (5 variants)
- start_agrisense.bat
- start_agrisense.ps1
- start_agrisense.py
- start.sh
- start_agrisense_scold.ps1
- start_optimized.ps1
- start_hybrid_ai.ps1
- **Action**: Consolidate to single entry point

#### 3. **Duplicate Training Scripts** (8+ variants)
- retrain_all_models_gpu.py
- retrain_fast_gpu.py
- retrain_fast_gpu.sh
- retrain_gpu.sh
- retrain_gpu_simple.py
- retrain_gpu_simple.sh
- retrain_production.py
- retrain_production.sh
- train_npu_models.bat
- train_npu_models.ps1
- **Action**: Consolidate to single training entry point

#### 4. **Duplicate Test Scripts** (4+ variants)
- test_gpu_backend.sh
- test_integration.ps1
- test_frontend_api_integration.ps1
- test_ml_endpoints.py
- test_ml_models_comprehensive.py
- **Action**: Consolidate to pytest framework

#### 5. **Obsolete Documentation** (~14,500 files)
- CLEANUP_*.md reports (outdated)
- OPTIMIZATION_*.md reports (various dates)
- ML_EVALUATION_*.md (evaluation reports)
- *_SUMMARY.md (duplicate summaries)
- Post-cleanup analysis reports
- **Action**: Archive to /documentation/archive/ or delete

#### 6. **Temporary & Log Files**
- gpu_training_20251228_110532.log
- training_output.log
- tmp_integration_test.py
- tmp_metrics_test.py
- tmp_obs_test.py
- temp_model.onnx.data
- Metadata: .file_sizes.json, .sizes_summary.json, .pip_freeze.txt
- **Action**: DELETE

#### 7. **Unused Dependencies**
- Possibly unused: tensorflow, torch, cuda-related, scold, phi
- **Note**: These are in requirements-ml.txt (optional)
- **Action**: Document which are truly unused

---

## 🎯 Cleanup Strategy

### Phase 1: Safe Deletions (No Risk)
```
✓ Delete virtual environments
  - venv312/
  - venv_ml312/
  - venv_npu/
  - .venv/
  - .venv312/
  - .venv.bak/

✓ Delete temporary files
  - tmp_*.py
  - *.log files (except development logs to keep)
  - temp_*.onnx.data
  - .file_sizes.json
  - .sizes_summary.json
  - .pip_freeze.txt

✓ Delete cleanup reports
  - cleanup_backup_20251205_182237/
  - CLEANUP_*.md
  - CLEANUP_*.txt
  - cleanup_optimize_project.ps1
  - fix_integration.ps1
```

**Recoverable Space**: ~5.2 GB

### Phase 2: Script Consolidation
```
✓ Consolidate startup scripts into: scripts/start.py
  - Replace all start_*.ps1, start_*.bat, start_*.sh
  - Add option flags for different modes

✓ Consolidate training scripts into: scripts/train.py
  - Replace retrain_*.py, train_*.ps1, train_*.bat
  - Add flags for: --gpu, --npu, --fast, --production

✓ Consolidate test scripts into: tests/ with pytest
  - Move to pytest framework
  - Replace test_*.ps1, test_*.sh variants
```

### Phase 3: Documentation Cleanup
```
✓ Keep only:
  - README.md (main project readme)
  - DOCUMENTATION_INDEX.md (reference)
  - ARCHITECTURE_DIAGRAM.md (system design)
  - E2E_TESTING_GUIDE.md (testing docs)
  - Production-related guides

✓ Archive to /documentation/archived/:
  - All *_REPORT.md files
  - All *_SUMMARY.md (except critical)
  - All OPTIMIZATION_*.md
  - All CLEANUP_*.md
  - All ML_EVALUATION_*.md
  - GPU_TRAINING_SESSION_SUMMARY.md
  - NPU_TRAINING_SESSION_SUMMARY.md
```

### Phase 4: Update Core Documentation
```
✓ Update README.md
  - Remove references to deleted scripts
  - Consolidate setup instructions
  - Point to scripts/start.py
  - Update requirements documentation

✓ Create DEPLOYMENT.md
  - Single source of truth for deployment
  - Link to Azure deployment guide

✓ Create DEVELOPMENT.md
  - Development environment setup
  - How to use consolidated scripts
  - Testing instructions
```

---

## 📝 Files to DELETE

### Virtual Environments (5.1 GB total)
```
venv312/
venv_ml312/
venv_npu/
.venv/
.venv312/
.venv.bak/
```

### Temporary Files
```
tmp_integration_test.py
tmp_metrics_test.py
tmp_obs_test.py
gpu_training_20251228_110532.log
training_output.log
temp_model.onnx.data
.file_sizes.json
.sizes_summary.json
.pip_freeze.txt
```

### Cleanup-Related Files
```
cleanup_backup_20251205_182237/
cleanup_optimize_project.ps1
comprehensive_cleanup.ps1
fix_integration.ps1
CLEANUP_COMPLETE_REPORT_20251224.md
CLEANUP_REPORT_20251205_182951.md
CLEANUP_SUMMARY_QUICK.md
DEPLOYMENT_CLEANUP_REPORT.md
POST_CLEANUP_ANALYSIS.md
CRITICAL_FIXES_REPORT.md
```

### Duplicate/Obsolete Reports
```
OPTIMIZATION_IMPLEMENTATION_COMPLETE.md
OPTIMIZATION_IMPLEMENTATION_GUIDE.md
OPTIMIZATION_QUICK_REFERENCE.md
OPTIMIZATION_SUMMARY.md
PRODUCTION_OPTIMIZATION_COMPLETE.md
PRODUCTION_OPTIMIZATION_IMPLEMENTATION_GUIDE.md
ML_EVALUATION_FINAL_SUMMARY.txt
ML_MODEL_EVALUATION_COMPREHENSIVE_REPORT.md
GPU_TRAINING_SESSION_SUMMARY.md
NPU_TRAINING_SESSION_SUMMARY.md
PYTHON_312_OPTIMIZATION_REPORT.md
ML_MODELS_QUICK_REFERENCE.md
NPU_IMPLEMENTATION_OVERVIEW.md
INTEGRATION_FIX_SUMMARY.md
INTEGRATION_FIXES_SUMMARY.md
FINAL_VALIDATION_REPORT.md
TROUBLESHOOTING_COMPLETE_REPORT.md
COMPREHENSIVE_PROJECT_EVALUATION.md
COMPREHENSIVE_OPTIMIZATION_REPORT.md
PROJECT_CLEANUP_PLAN.md
PROJECT_ANALYSIS_COMPLETE.md
RETRAINING_COMPLETE.md
```

### Duplicate Startup Scripts (Keep only scripts/start.py)
```
start_agrisense.bat
start_agrisense.ps1
start_agrisense.py
start.sh
start_agrisense_scold.ps1
start_optimized.ps1
start_hybrid_ai.ps1
```

### Duplicate Training Scripts (Keep only scripts/train.py)
```
retrain_all_models_gpu.py
retrain_fast_gpu.py
retrain_fast_gpu.sh
retrain_gpu.sh
retrain_gpu_simple.py
retrain_gpu_simple.sh
retrain_production.py
retrain_production.sh
train_npu_models.bat
train_npu_models.ps1
```

### Installation Scripts (Keep standardized approach)
```
install_cuda_wsl2.bat
install_cuda_wsl2.ps1
install_cuda_wsl2.sh
setup_npu_environment.ps1
setup_phi_scold.ps1
```

---

## ✅ Files to KEEP & Update

### Core Documentation
```
README.md - UPDATE with latest changes
DOCUMENTATION_INDEX.md - UPDATE to reflect current structure
ARCHITECTURE_DIAGRAM.md - VERIFY still accurate
ENV_VARS_REFERENCE.md - UPDATE if needed
E2E_TESTING_GUIDE.md - UPDATE test instructions
```

### New Documentation to Create
```
DEPLOYMENT.md - Consolidated deployment guide
DEVELOPMENT.md - Development setup & workflow
QUICKSTART.md - Quick start for new developers
TESTING.md - Testing strategy and procedures
```

### Configuration Files to Keep
```
.gitignore - UPDATE to ignore venv/
.env.example
.env.production.template
pytest.ini
tsconfig.json
playwright.config.ts
```

### Reference Files
```
E2E_ANALYSIS_REPORT.json - KEEP (newly generated)
analysis_report.json
ML_MODEL_TEST_RESULTS.json
npu_benchmark_results.json
retraining_report_20260102_123209.json
HARDWARE_OPTIMIZATION_CONFIG.md
PYTHON_312_UPGRADE_SUMMARY.md
PYTHON_312_QUICK_REFERENCE.md
CUDA_QUICK_START.md
NPU_QUICK_START.md
WSL2_CUDA_SETUP_GUIDE.md
SCOLD_QUICK_START.md
```

---

## 🔧 Script Consolidation Templates

### 1. scripts/start.py (Single Startup Entry Point)
```python
#!/usr/bin/env python3
"""
AgriSense unified startup script
Usage: python scripts/start.py [--backend] [--frontend] [--all] [--dev] [--prod]
"""
# Replaces: start_agrisense.bat, start_agrisense.ps1, start_agrisense.py, start.sh, etc.
```

### 2. scripts/train.py (Single Training Entry Point)
```python
#!/usr/bin/env python3
"""
AgriSense model training script
Usage: python scripts/train.py [--gpu] [--npu] [--fast] [--production]
"""
# Replaces: retrain_*.py, train_*.ps1, train_*.bat
```

### 3. tests/conftest.py (Unified pytest Configuration)
```
# Consolidate all test configurations
# Replace: test_*.ps1, test_*.sh, test_*.py variants
```

---

## 📋 Updated .gitignore
```
# Virtual Environments
venv/
venv312/
venv_ml312/
venv_npu/
.venv/
.venv312/
.venv.bak/

# IDE & Editor
.vscode/
.idea/
*.swp
*.swo

# Build & Dist
build/
dist/
*.egg-info/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.pytest_cache/

# Node
node_modules/
npm-debug.log
yarn-error.log

# Environment Variables
.env.local
.env.*.local

# Temp Files
tmp_*.py
*.log
temp_*.onnx*
```

---

## 🚀 Execution Steps

### Step 1: Backup (IMPORTANT!)
```bash
git checkout main  # Ensure on main branch
git pull           # Get latest code
# Create backup branch
git checkout -b cleanup/e2e-2026-01-03
```

### Step 2: Delete Virtual Environments
```bash
rm -rf venv312 venv_ml312 venv_npu .venv .venv312 .venv.bak
```

### Step 3: Delete Temporary Files
```bash
rm -f tmp_*.py *.log temp_*.onnx.data
rm -f .file_sizes.json .sizes_summary.json .pip_freeze.txt
```

### Step 4: Delete Obsolete Reports
```bash
rm -f CLEANUP_*.md OPTIMIZATION_*.md ML_EVALUATION_*.md
rm -rf cleanup_backup_*
```

### Step 5: Consolidate Scripts
- Move duplicate startup scripts → scripts/start.py
- Move duplicate training scripts → scripts/train.py  
- Move test files → tests/ with pytest

### Step 6: Update Documentation
- README.md - Update with new structure
- Create DEPLOYMENT.md, DEVELOPMENT.md, QUICKSTART.md
- Archive old reports to /documentation/archived/

### Step 7: Verify & Commit
```bash
pytest tests/  # Run all tests
python scripts/start.py --help  # Test new scripts
git add -A
git commit -m "E2E cleanup: Remove venvs, consolidate scripts, update docs"
git push origin cleanup/e2e-2026-01-03
```

---

## 💾 Space Recovery Summary

| Item | Size | Status |
|------|------|--------|
| Virtual Environments | 5.1 GB | DELETE |
| Temporary Files | ~50 MB | DELETE |
| Obsolete Reports | ~100 MB | ARCHIVE/DELETE |
| **Total Recoverable** | **~5.2 GB** | **ACTION REQUIRED** |

---

## ⚠️ Important Notes

1. **Virtual Environments**: Should NEVER be committed to git. Update `.gitignore`
2. **Dependencies**: Use `requirements.txt` (core) + `requirements-ml.txt` (optional)
3. **Backward Compatibility**: Ensure new consolidated scripts support all old flags
4. **Testing**: Run full test suite before merging cleanup
5. **Documentation**: Update all docs to reference new script locations

---

## 📅 Next Steps

1. ✓ Analysis complete
2. → Execute cleanup (Phase 1-4)
3. → Update documentation  
4. → Run test suite
5. → Create pull request
6. → Merge after review

