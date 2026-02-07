# Claude Crop Recommender - Troubleshooting Guide

## Quick Diagnosis

**Start here to quickly identify your issue:**

### Symptom 1: Cannot import crop recommender
```
ModuleNotFoundError: No module named 'claude_crop_recommender'
```
**→ Go to:** [Missing Dependencies](#missing-dependencies)

### Symptom 2: Model training fails
```
ValueError: Could not load crop dataset
```
**→ Go to:** [Model Training Issues](#model-training-issues)

### Symptom 3: Python service not running
```
ConnectionError: HTTPConnectionPool(host='localhost', port=8000)
```
**→ Go to:** [Python Service Connection](#python-service-connection)

### Symptom 4: Predictions return wrong crops
```
Got "Carrot" for rice fields!
```
**→ Go to:** [Model Accuracy Issues](#model-accuracy-issues)

### Symptom 5: Slow predictions (>5 seconds)
```
Request times out waiting for response
```
**→ Go to:** [Performance Issues](#performance-issues)

---

## Missing Dependencies

### Issue: `ModuleNotFoundError: No module named 'pandas'`

**Cause:** Python packages not installed

**Solution:**
```bash
cd backend/ml/claude_crop_recommender
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import pandas; import numpy; import sklearn; print('✓ All packages installed')"
```

**If still failing:**
```bash
# Option 1: Clear pip cache and reinstall
pip install --force-reinstall -r requirements.txt

# Option 2: Use specific versions
pip install pandas==1.3.5 numpy==1.21.0 scikit-learn==0.24.2 joblib==1.0.1

# Option 3: Check Python version
python --version  # Must be 3.8+
```

### Issue: `ModuleNotFoundError: No module named 'backend'`

**Cause:** Running Python from wrong directory

**Solution:**
```bash
# Wrong:
cd backend/ml/claude_crop_recommender && python crop_recommender_api.py

# Correct:
cd backend && python -m ml.claude_crop_recommender.crop_recommender_api
```

Or use FastAPI directly:
```bash
cd backend
python -m uvicorn ai_service.main:app --port 8000
```

### Issue: `ImportError: cannot import name 'router' from routes`

**Cause:** Routes file not properly created or imported

**Solution:**
1. **Verify file exists:** `ls backend/ml/claude_crop_recommender/routes.py`
2. **Check syntax:** `python -m py_compile backend/ml/claude_crop_recommender/routes.py`
3. **Verify __init__.py:** `cat backend/ml/claude_crop_recommender/__init__.py`

**Expected __init__.py:**
```python
from .crop_recommendation_ml_model import CropRecommendationSystem
from .crop_recommender_api import recommend_crops, SoilParameters

__all__ = ['CropRecommendationSystem', 'recommend_crops', 'SoilParameters']
```

---

## Model Training Issues

### Issue: "Model training started..." but never completes

**Cause:** Training takes longer than expected (2-3 minutes is normal)

**Solution:**
1. Wait 3-5 minutes before stopping
2. Check if file is being created: `ls -lh backend/ml/claude_crop_recommender/crop_recommendation_model.pkl`
3. Monitor memory: `free -h` (on Linux) or Task Manager (on Windows)

**If memory is full:**
```bash
# Reduce training samples per crop
# Edit crop_recommendation_ml_model.py, line ~120:
# Change: samples_per_crop=50
# To: samples_per_crop=30
```

### Issue: "ValueError: Could not load crop dataset"

**Cause:** crop_requirements_dataset.py file missing or corrupted

**Solution:**
1. **Verify file exists:** `ls backend/ml/claude_crop_recommender/crop_requirements_dataset.py`
2. **Check file size:** Should be ~15KB (text) when uncompressed
3. **Verify syntax:** `python -m py_compile backend/ml/claude_crop_recommender/crop_requirements_dataset.py`

**If file is corrupted:**
- Delete it: `rm backend/ml/claude_crop_recommender/crop_requirements_dataset.py`
- Re-create from template in INTEGRATION_SUMMARY.md
- Or re-download from project

### Issue: "Model accuracy is below 70%"

**Cause:** Training data generation issue

**Solution:**
```bash
# Retrain with more data
python -c "
from backend.ml.claude_crop_recommender.crop_recommender_api import get_crop_recommender
from backend.ml.claude_crop_recommender.crop_requirements_dataset import crop_data
import pandas as pd

recommender = get_crop_recommender()
crop_df = pd.DataFrame(crop_data)

# Change samples_per_crop to 100 for better accuracy
recommender.generate_training_data(crop_df, samples_per_crop=100)
recommender.train(crop_df)
print(f'Training complete. Best model: {recommender.best_model_name}')
"
```

### Issue: "MemoryError: Unable to allocate"

**Cause:** Training uses too much RAM (2.5GB peak)

**Solution:**
```bash
# Option 1: Reduce training samples
# Edit crop_recommendation_ml_model.py:
# samples_per_crop=50 → samples_per_crop=20

# Option 2: Close other applications to free RAM

# Option 3: Use smaller models only (disable Neural Network)
# Edit crop_recommender_api.py, comment out NN model training
```

### Issue: "FileNotFoundError: crop_recommendation_model.pkl"

**Cause:** Model file not generated

**Solution:**
```bash
# Generate model
cd backend
python -m ai_service.main:app --startup-event

# OR manually train
python -c "
from ml.claude_crop_recommender.crop_recommender_api import get_crop_recommender
recommender = get_crop_recommender()
print('Model generated at: backend/ml/claude_crop_recommender/crop_recommendation_model.pkl')
"

# Verify
ls -lh backend/ml/claude_crop_recommender/crop_recommendation_model.pkl
```

---

## Python Service Connection

### Issue: "ConnectionError: HTTPConnectionPool(host='localhost', port=8000)"

**Cause:** Python FastAPI service not running

**Solution:**
```bash
# Check if service is running
curl http://localhost:8000/crop-recommendation/health

# If not running, start it
cd backend
python -m uvicorn ai_service.main:app --host 0.0.0.0 --port 8000

# Wait for:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
```

**Alternative - Check port in use:**
```bash
# On Windows
netstat -ano | findstr :8000

# On Mac/Linux
lsof -i :8000

# Kill process on that port
kill -9 <PID>  # Or taskkill /PID <PID> /F on Windows
```

### Issue: "Service running but returns 404"

**Cause:** Routes not properly registered with FastAPI

**Solution:**
1. **Verify routes.py:**
```bash
python -c "
from backend.ml.claude_crop_recommender.routes import router
print(f'Router has {len(router.routes)} routes')
print([route.path for route in router.routes])
"
```

2. **Verify main.py includes router:**
```python
# Check backend/ai_service/main.py contains:
from backend.ml.claude_crop_recommender.routes import router as crop_recommendation_router
app.include_router(crop_recommendation_router)
```

3. **Restart service:**
```bash
# Stop service (Ctrl+C)
# Start again with reload
python -m uvicorn ai_service.main:app --reload --port 8000
```

### Issue: "HTTP 500 error on /crop-recommendation/predict"

**Cause:** Server-side error during prediction

**Solution:**
```bash
# Check error details in terminal where service is running
# Look for exception traceback

# Try basic health check first
curl http://localhost:8000/crop-recommendation/health

# Test with valid parameters
curl -X POST http://localhost:8000/crop-recommendation/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pH": 6.5,
    "N": 100,
    "P": 50,
    "K": 150,
    "Fe": 4.06,
    "Mn": 1.68,
    "Zn": 0.83,
    "Cu": 0.46,
    "B": 0.3,
    "Water": 500,
    "Moisture": 60,
    "Temperature": 25,
    "Rainfall": 700
  }'

# Check logs for specific error message
```

### Issue: "Timeout: Service took too long to respond"

**Cause:** 
1. Model training happening (first request)
2. Service is overloaded
3. Network issue

**Solution:**
```bash
# Check if model is still training
du -h backend/ml/claude_crop_recommender/crop_recommendation_model.pkl

# Increase timeout in Express controller
# Edit: backend/controllers/cropRecommendationController.js
# Change: timeout: 30000
# To: timeout: 60000

# Or increase FastAPI timeout
python -m uvicorn ai_service.main:app --timeout-keep-alive 60
```

---

## Express Backend Issues

### Issue: "EADDRINUSE: address already in use :::3000"

**Cause:** Another process using port 3000

**Solution:**
```bash
# On Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# On Mac/Linux
lsof -i :3000
kill -9 <PID>

# Or use different port
npm start -- --port 3001
```

### Issue: Express not calling Python service

**Cause:** PYTHON_SERVICE_URL not set or wrong

**Solution:**
```bash
# Check env variable
echo $PYTHON_SERVICE_URL  # On Linux/Mac
echo %PYTHON_SERVICE_URL%  # On Windows

# Set env variable
export PYTHON_SERVICE_URL=http://localhost:8000  # Linux/Mac
set PYTHON_SERVICE_URL=http://localhost:8000     # Windows

# Or add to .env file
echo "PYTHON_SERVICE_URL=http://localhost:8000" >> backend/.env

# Restart Node service
npm start
```

### Issue: "TypeError: Cannot read property 'crop' of undefined"

**Cause:** Response format mismatch

**Solution:**
1. **Check response format:**
```bash
curl -X POST http://localhost:3000/api/crop-recommendation/predict \
  -H "Content-Type: application/json" \
  -d '{"N": 120, "P": 54, "K": 100, "temperature": 28, "humidity": 60, "ph": 7.0, "rainfall": 600}'
```

2. **Verify response includes primary_recommendation:**
```json
{
  "data": {
    "primary_recommendation": {
      "crop": "... should have this
    }
  }
}
```

3. **Check cropRecommendationService.js mapping functions** are present

---

## Frontend Issues

### Issue: React component shows "undefined" for recommendations

**Cause:** API response not being displayed correctly

**Solution:**
```typescript
// In your component, verify structure
useEffect(() => {
  if (result?.recommendations) {
    console.log('Recommendations:', result.recommendations);
  } else if (result?.all_recommendations) {
    console.log('All recommendations:', result.all_recommendations);
  } else {
    console.log('Result structure:', result);
  }
}, [result]);
```

### Issue: CORS error when calling API

**Cause:** Frontend and backend ports mismatch

**Solution:**
```bash
# Check port frontend is running on
# Should match API_BASE in frontend/services/api.ts

# For development (typical):
# Frontend: http://localhost:5173 (Vite)
# Backend: http://localhost:3000 (Express)

# Update in frontend/services/api.ts:
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:3000';
```

### Issue: Recommendations don't seem reasonable

**Cause:** New UI component not adapted for new response format

**Solution:**
```typescript
// Old format handling (backward compat)
if (result.crop === 'Rice') { /* ... */ }

// New format handling (enhanced)
if (result.recommendations?.[0].crop === 'Rice') { /* ... */ }

// Safe approach (supports both)
const topCrop = result.recommendations?.[0]?.crop || result.crop;
```

---

## Model Accuracy Issues

### Issue: "Model recommends wrong crop for given parameters"

**Cause:** 
1. Model not trained well
2. Input parameters out of range
3. Crop parameters dataset incomplete

**Solution:**
```bash
# 1. Check model accuracy
python -c "
from backend.ml.claude_crop_recommender.crop_recommender_api import get_crop_recommender
r = get_crop_recommender()
print(f'Model accuracy: {r.test_accuracy:.2%}')
print(f'Best model: {r.best_model_name}')
"

# 2. Verify crop parameters
python -c "
from backend.ml.claude_crop_recommender.crop_requirements_dataset import crop_data
import json
for crop, params in list(crop_data.items())[:3]:
    print(f'{crop}: {params}')
"

# 3. Test with known good parameters
# Rice typically needs: pH 6.5-7.0, N 100-150, P 40-60, K 40-60
curl -X POST http://localhost:8000/crop-recommendation/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pH": 6.8,
    "N": 120,
    "P": 50,
    "K": 50,
    "Fe": 4.06,
    "Mn": 1.68,
    "Zn": 0.83,
    "Cu": 0.46,
    "B": 0.3,
    "Water": 500,
    "Moisture": 60,
    "Temperature": 28,
    "Rainfall": 600
  }'
```

### Issue: "Confidence scores are all very low (<50%)"

**Cause:** 
1. Input parameters far from any crop's requirements
2. Model needs retraining
3. Input feature scaling issue

**Solution:**
```bash
# 1. Check parameter ranges
# Verify inputs are within reasonable ranges:
# pH: 4.5-8.5
# N: 10-300
# Temperature: 10-40°C
# Rainfall: 0-500mm/month

# 2. Retrain with more data
python -c "
from backend.ml.claude_crop_recommender.crop_recommender_api import get_crop_recommender
from backend.ml.claude_crop_recommender.crop_requirements_dataset import crop_data
import pandas as pd

r = get_crop_recommender()
crop_df = pd.DataFrame(crop_data)
r.train(crop_df)  # Force retrain
print('Training complete')
"

# 3. Check if parameters are being normalized correctly
# Log input values in crop_recommender_api.py recommend_crops()
```

---

## Performance Issues

### Issue: "First prediction takes 3+ minutes"

**Cause:** Model training on first request (expected)

**Solution:**
```bash
# Pre-train model before starting service
cd backend
python -c "
from ml.claude_crop_recommender.crop_recommender_api import get_crop_recommender
print('Pre-training model...')
r = get_crop_recommender()
print(f'Model ready! Accuracy: {r.test_accuracy:.2%}')
"

# Then start service - subsequent predictions will be fast
python -m uvicorn ai_service.main:app --port 8000
```

### Issue: "Subsequent predictions are slow (>500ms)"

**Cause:** 
1. Large dataset queries
2. Inefficient model serialization
3. Network overhead

**Solution:**
```bash
# 1. Check if model is cached
python -c "
from backend.ml.claude_crop_recommender.crop_recommender_api import _crop_recommender
print(f'Model is cached: {_crop_recommender is not None}')
"

# 2. Profile the prediction
import time
start = time.time()
# Make prediction
elapsed = time.time() - start
print(f'Elapsed: {elapsed:.3f}s')

# 3. If still slow, optimize in crop_recommendation_ml_model.py
# Use smaller datasets or SimpleML models
```

### Issue: "Memory usage grows over time (memory leak)"

**Cause:** Model instances not being cleaned up

**Solution:**
```python
# Verify singleton pattern is used correctly in routes.py
# Should have:

_crop_recommender = None

def get_crop_recommender():
    global _crop_recommender
    if _crop_recommender is None:
        _crop_recommender = CropRecommendationSystem()
        # Train only once
    return _crop_recommender
```

---

## Data Issues

### Issue: "Input validation failures (400 Bad Request)"

**Cause:** Parameters outside valid range

**Solution:**
```bash
# Check valid ranges
curl http://localhost:8000/crop-recommendation/crop-requirements/Rice

# Expected response shows ranges:
{
  "pH_Min": 5.5,
  "pH_Max": 7.0,
  "N_Min": 80,
  "N_Max": 150,
  ...
}

# Adjust input to fit ranges
```

### Issue: "Crop not found in dataset"

**Cause:** Typo in crop name or crop not in database

**Solution:**
```bash
# Get list of all crops
curl http://localhost:8000/crop-recommendation/crops-list

# Use exact names from response
# Examples: "Rice" not "rice", "Wheat" not "wheat"
```

---

## Logging & Debugging

### Enable Debug Logging

**Python (FastAPI):**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("CropRecommendation")
logger.debug("Starting prediction...")
```

**Node.js:**
```javascript
const debug = require('debug')('cropRecommendation');
debug('Calling Python service...');
```

### Check Service Logs

```bash
# View Python service logs (in terminal)
# Look for [INFO], [WARNING], [ERROR] messages

# View Express logs
tail -f backend/logs/app.log

# View complete request/response
curl -v http://localhost:3000/api/crop-recommendation/predict
```

### Save Debug Information

```bash
# Save request/response to file
curl -X POST http://localhost:3000/api/crop-recommendation/predict \
  -H "Content-Type: application/json" \
  -d '{...}' \
  > debug_response.json 2>&1

# View with formatting
cat debug_response.json | python -m json.tool
```

---

## Common Error Messages & Solutions

| Error Message | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Missing Python package | `pip install -r requirements.txt` |
| `ConnectionError: HTTPConnectionPool` | Python service not running | Start FastAPI service on port 8000 |
| `HTTP 422 Unprocessable Entity` | Invalid request format | Check parameter types and ranges |
| `HTTP 500 Internal Server Error` | Server-side error | Check terminal logs for traceback |
| `Timeout: Service took too long` | Slow response | Model may be training (normal first time) |
| `CORS error: No 'Access-Control-Allow-Origin'` | Frontend-backend mismatch | Check CORS configuration |
| `TypeError: Cannot read property` | Response structure wrong | Verify response includes all fields |
| `MemoryError` | Not enough RAM for training | Reduce `samples_per_crop` or close apps |

---

## Verification Checklist

Use this after troubleshooting to verify everything works:

- [ ] Python packages installed: `pip list | grep pandas`
- [ ] Model file exists: `ls -lh backend/ml/claude_crop_recommender/crop_recommendation_model.pkl`
- [ ] Python service running: `curl http://localhost:8000/crop-recommendation/health`
- [ ] Health check passes: Response includes `"status": "ok"`
- [ ] Express backend running: `curl http://localhost:3000/health`
- [ ] Can make prediction: `curl -X POST http://localhost:3000/api/crop-recommendation/predict ...`
- [ ] Response has correct format: Includes `primary_recommendation`, `all_recommendations`, `analysis`
- [ ] Fallback works: Stop Python service, try prediction again (should still work)

---

## Still Stuck?

If none of the above helps:

1. **Check all logs** in both terminals (Python and Node)
2. **Verify all files** are present in correct locations
3. **Test each layer separately:**
   - Python service: `curl http://localhost:8000/docs`
   - Express backend: `curl http://localhost:3000/health`
   - Frontend: Check browser console (F12)
4. **Re-install dependencies:** `pip install --force-reinstall -r requirements.txt`
5. **Check file permissions:** Ensure readable/writable
6. **Restart everything** (completely close terminals and restart)

---

**Last Updated:** February 7, 2026  
**Status:** Production Troubleshooting Guide Complete
