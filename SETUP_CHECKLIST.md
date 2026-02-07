# Claude Crop Recommender - Setup Checklist

## Pre-Integration Setup (Do First)

- [ ] **Clone/Download** the Claude crop recommender files from Google Drive
- [ ] **Verify** Python 3.8+ is installed: `python --version`
- [ ] **Verify** pip is working: `pip --version`

## Step 1: Install Python Dependencies (5 minutes)

```bash
cd f:\AGRISENSEFULL-STACK\backend\ml\claude_crop_recommender
pip install -r requirements.txt
```

**Expected Output:**
```
Successfully installed pandas numpy scikit-learn joblib
```

**Troubleshooting:**
- If you get permission error, use: `pip install --user -r requirements.txt`
- If you get version conflict, create a venv: `python -m venv venv` then activate

## Step 2: Initialize the Model (10 minutes - First Time Only)

```bash
python crop_recommender_api.py
```

**Expected Output:**
```
2024-XX-XX 10:30:00 - Model training started...
2024-XX-XX 10:32:45 - Training completed!
2024-XX-XX 10:32:46 - Random Forest model accuracy: 92.5%
Model saved to crop_recommendation_model.pkl
```

**First run will be slow (~2-3 minutes).** Subsequent runs use cached model (fast).

**Troubleshooting:**
- If training fails, check you installed all dependencies correctly
- If disk space error, ensure 100MB free space in `backend/ml/claude_crop_recommender/`

## Step 3: Integrate FastAPI Routes (2 minutes)

**Edit:** `backend/ai_service/main.py`

**Add these lines** near the top of the file (after other imports):
```python
from backend.ml.claude_crop_recommender.routes import router as crop_recommendation_router
```

**Add this line** after `app = FastAPI(...)` initialization:
```python
app.include_router(crop_recommendation_router)
```

**Example of where to add:**
```python
# At the top
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.ml.claude_crop_recommender.routes import router as crop_recommendation_router  # ← NEW

# Define app
app = FastAPI(title="AgriSense API")
app.include_router(crop_recommendation_router)  # ← NEW
```

**Verify:** Save file, no syntax errors

## Step 4: Configure Environment (2 minutes)

**Create or edit:** `backend/.env`

```env
# Existing variables...

# NEW: Python ML Service
PYTHON_SERVICE_URL=http://localhost:8000
USE_ML_SERVICE=true
```

**Verify:** File saved and contains both variables

## Step 5: Start Services (Depends)

### Option A: Quick Test (Use FastAPI Only)

**Terminal 1:**
```bash
cd backend
python -m uvicorn ai_service.main:app --host 0.0.0.0 --port 8000 --reload
```

Wait for:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**In another Terminal:**
```bash
# Test the service
curl http://localhost:8000/crop-recommendation/health

# Expected response:
{"status":"ok","crops_supported":100,"model":"Random Forest"}
```

### Option B: Full Stack (FastAPI + Express Backend)

**Terminal 1 - Python Service:**
```bash
cd backend
python -m uvicorn ai_service.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Node Backend:**
```bash
cd backend
npm install  # First time only
npm start
```

**Terminal 3 - React Frontend (Optional):**
```bash
cd frontend
npm install  # First time only
npm run dev
```

## Step 6: Test Integration (5 minutes)

### Test 1: Python Service Health
```bash
curl http://localhost:8000/crop-recommendation/health
```

**Expected Response:**
```json
{
  "status": "ok",
  "crops_supported": 100,
  "model": "Random Forest",
  "accuracy": 0.92
}
```

### Test 2: Get Available Crops
```bash
curl http://localhost:8000/crop-recommendation/crops-list
```

**Expected Response:**
```json
{
  "total_crops": 100,
  "categories": {
    "Cereals": 15,
    "Pulses": 15,
    "Oilseeds": 15,
    ...
  },
  "crops": ["Rice", "Wheat", "Maize", ...]
}
```

### Test 3: Make a Prediction (Via Python)
```bash
curl -X POST http://localhost:8000/crop-recommendation/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pH": 7.0,
    "N": 120,
    "P": 54,
    "K": 100,
    "Fe": 4.06,
    "Mn": 1.68,
    "Zn": 0.83,
    "Cu": 0.46,
    "B": 0.3,
    "Water": 500,
    "Moisture": 60,
    "Temperature": 28,
    "Rainfall": 600,
    "top_n": 5
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "rank": 1,
      "crop_name": "Rice",
      "suitability_score": 92.5,
      "confidence": 0.95
    },
    ...
  ],
  "model_info": {
    "name": "Random Forest",
    "accuracy": 0.92,
    "total_crops": 100
  }
}
```

### Test 4: Via Express Backend
```bash
curl -X POST http://localhost:3000/api/crop-recommendation/predict \
  -H "Content-Type: application/json" \
  -d '{
    "N": 120,
    "P": 54,
    "K": 100,
    "temperature": 28,
    "humidity": 60,
    "ph": 7.0,
    "rainfall": 600
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "data": {
    "primary_recommendation": {
      "crop": "Rice",
      "confidence": 0.92,
      "suitability_score": "92.50"
    },
    "alternatives": ["Wheat", "Maize"],
    "all_recommendations": [...],
    "analysis": {
      "soil_status": {...},
      "climate_status": {...}
    }
  }
}
```

### Test 5: Test Fallback (Stop Python Service & Try Again)

1. **Stop the Python service** (Ctrl+C in Terminal 1)
2. **Make request to Express again:**
```bash
curl -X POST http://localhost:3000/api/crop-recommendation/predict \
  -H "Content-Type: application/json" \
  -d '{"N": 120, "P": 54, "K": 100, "temperature": 28, "humidity": 60, "ph": 7.0, "rainfall": 600}'
```

**Expected:** Still works! (Falls back to native ML or rules)

## Step 7: View Documentation

**Python API Docs (Swagger):**
```
http://localhost:8000/docs
```

**Backend Docs:**
Open: `backend/ml/claude_crop_recommender/README.md`

**Integration Guide:**
Open: `CLAUDE_INTEGRATION_GUIDE.md` (root folder)

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'pandas'"
**Solution:** 
```bash
pip install -r backend/ml/claude_crop_recommender/requirements.txt
```

### Issue: "Connection refused" when calling Python service
**Solution:**
- Check if Python FastAPI is running: `curl http://localhost:8000/docs`
- If not running, start it: `python -m uvicorn ai_service.main:app --port 8000`
- Verify PYTHON_SERVICE_URL env var is correct

### Issue: Model training takes too long
**Solution:**
- First run is slow (~2-3 minutes). This is normal.
- Subsequent runs use cached model (100ms)
- If it's stuck, press Ctrl+C and check logs

### Issue: "Permission denied" when installing
**Solution:**
```bash
pip install --user -r requirements.txt
```

## Verification Checklist

- [ ] Python dependencies installed: `pip list | grep pandas`
- [ ] Model trained: `ls backend/ml/claude_crop_recommender/crop_recommendation_model.pkl`
- [ ] FastAPI routes added to `main.py`
- [ ] Environment variables set in `.env`
- [ ] Python service running: `curl http://localhost:8000/docs`
- [ ] Health check passes: `curl http://localhost:8000/crop-recommendation/health`
- [ ] Prediction works: Made test request and got valid response
- [ ] Fallback works: Still works when Python service is down

## Next Steps

1. ✅ **Complete setup** (all steps above)
2. ⬜ **Update frontend UI** to display new recommendation format (optional but recommended)
3. ⬜ **Add monitoring/logging** for production use
4. ⬜ **Run full test suite** to ensure no regressions
5. ⬜ **Deploy to staging** for real-world testing
6. ⬜ **Deploy to production** when satisfied

## Need Help?

1. **Check documentation**: `CLAUDE_INTEGRATION_GUIDE.md`
2. **Check README**: `backend/ml/claude_crop_recommender/README.md`
3. **View API docs**: http://localhost:8000/docs (when service running)
4. **Check logs**: Look for errors in terminal output
5. **Test endpoints**: Use curl commands above

## Quick Commands Reference

```bash
# Start Python service
python -m uvicorn backend.ai_service.main:app --port 8000

# Start Node backend
cd backend && npm start

# Install dependencies
pip install -r backend/ml/claude_crop_recommender/requirements.txt

# Check service health
curl http://localhost:8000/crop-recommendation/health

# View API docs
open http://localhost:8000/docs

# Make a prediction
curl -X POST http://localhost:8000/crop-recommendation/predict \
  -H "Content-Type: application/json" \
  -d '{"pH":7,"N":120,"P":54,"K":100,"Fe":4.06,"Mn":1.68,"Zn":0.83,"Cu":0.46,"B":0.3,"Water":500,"Moisture":60,"Temperature":28,"Rainfall":600}'
```

---

**Setup Time Estimate:** 15-30 minutes  
**Status:** ✅ Ready for Integration  
**Last Updated:** February 7, 2026
