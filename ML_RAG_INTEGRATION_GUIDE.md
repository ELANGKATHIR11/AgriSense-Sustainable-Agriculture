# AgriSense ML & RAG Integration Guide

## 🎯 What Was Implemented

### 1. **ML Model Training Pipeline**
- **5 Trained Models** on prepared agricultural datasets:
  - Crop Recommendation (96 classes)
  - Crop Type Classification (10 classes)
  - Growth Duration Prediction (Regression: 18-365 days)
  - Water Requirement Estimation (Regression: 2.5-15 mm/day)
  - Season Classification (5 classes: Kharif, Rabi, Zaid, Perennial)

- **Intent Classifier** for RAG (SVM-based)
  - Classifies user queries into 5 intents
  - Supports: Weather, Disease, Soil, Crop Recommendation, Pricing

### 2. **Retrieval-Augmented Generation (RAG) Pipeline**
Three-component hybrid system:

**Component 1: Intent Classification (SVM)**
```
User Query → Intent Classifier → 
- Weather Intent
- Disease Intent  
- Soil Intent
- Crop Recommendation Intent
- Pricing Intent
```

**Component 2: Retrieval (Cosine Similarity)**
- Uses SBERT-like embeddings on crop features
- Retrieves relevant crops from dataset
- Supports multi-criteria search

**Component 3: Generation**
- Natural language response generation
- Based on retrieved data + intent
- Contextual recommendations

### 3. **Backend API Integration**
New endpoints at `/api/v1/ml/`:

```
POST /ml/rag/query           → Process RAG queries
POST /ml/predict/crop-recommendation
POST /ml/predict/crop-type
POST /ml/predict/growth-duration
POST /ml/predict/water-requirement
POST /ml/predict/season
GET  /ml/crops/search        → Search by criteria
GET  /ml/crops/recommendations
GET  /ml/models/info         → Model information
GET  /ml/health              → Health check
```

### 4. **Frontend Chat Component**
React component with:
- Real-time RAG chat interface
- Context-aware queries (season, crop type)
- Intent classification display
- Confidence scores
- Recommendation cards
- Auto-scrolling chat

---

## 📦 Files Created/Modified

### Backend Files
```
backend/ml/train_models.py          (490 lines) - Model training pipeline
backend/ml/rag_pipeline.py          (400 lines) - RAG implementation
backend/ml/inference.py             (350 lines) - Inference utilities
backend/api/routes/ml_predictions.py (450 lines) - API endpoints

backend/ml/models/
├── crop_recommendation_model.pkl
├── crop_type_classification_model.pkl
├── growth_duration_model.pkl
├── water_requirement_model.pkl
├── season_classification_model.pkl
├── intent_classifier_model.pkl
├── intent_classifier_scaler.pkl
├── model_metrics.json
└── model_manifest.json
```

### Frontend Files
```
frontend/src/components/AgriSenseRAGChat.tsx (280 lines) - RAG chat UI
```

---

## 🚀 Integration Steps

### Step 1: Install Dependencies

```bash
# Backend dependencies
pip install scikit-learn pandas numpy fastapi pydantic

# Frontend dependencies  
npm install @tanstack/react-query axios
```

### Step 2: Include ML Routes in Main Backend App

**In `backend/main.py` or FastAPI app initialization:**

```python
from api.routes.ml_predictions import mount_ml_routes

# After creating FastAPI app
app = FastAPI()

# Mount ML routes
mount_ml_routes(app)

# Or manually include router:
from api.routes.ml_predictions import router
app.include_router(router)
```

### Step 3: Initialize RAG Pipeline on Startup

**In `backend/main.py` startup event:**

```python
from ml.rag_pipeline import initialize_rag_pipeline
from ml.inference import get_inference_engine

@app.on_event("startup")
async def startup_event():
    """Initialize ML components on startup"""
    try:
        # Initialize inference engine (loads all models)
        engine = get_inference_engine()
        print("✅ ML models loaded successfully")
        
        # Initialize RAG pipeline
        pipeline = initialize_rag_pipeline()
        print("✅ RAG pipeline initialized")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize ML components: {e}")
```

### Step 4: Add RAG Chat to Frontend

**In `frontend/src/pages/App.tsx` or router:**

```typescript
import AgriSenseRAGChat from '@/components/AgriSenseRAGChat';

// Add route or component
<Route path="/chat" element={<AgriSenseRAGChat />} />

// Or embed in a page
export function ChatPage() {
  return <AgriSenseRAGChat />;
}
```

### Step 5: Configure API Base URL

**In `frontend/.env`:**

```env
VITE_API_URL=http://localhost:8000
# or for production
VITE_API_URL=https://api.agrisense.com
```

---

## 🧪 Testing the Integration

### Test 1: Check ML Health

```bash
curl http://localhost:8000/api/v1/ml/health
```

Expected response:
```json
{
  "status": "ready",
  "models_count": 5,
  "models": "crop_recommendation, crop_type_classification, growth_duration, water_requirement, season_classification"
}
```

### Test 2: Test RAG Query

```bash
curl -X POST http://localhost:8000/api/v1/ml/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What crops should I plant in Kharif season?",
    "season": "Kharif",
    "crop_type": ""
  }'
```

### Test 3: Test Crop Recommendation

```bash
curl -X POST http://localhost:8000/api/v1/ml/predict/crop-recommendation \
  -H "Content-Type: application/json" \
  -d '{
    "crop_name": "Test",
    "features": [25, 32, 6.5, 7.0, 5.0, 600, 1000, 70, 90, 120, 60, 40, 0.8, 80, 40, 30, 0, 100, 0]
  }'
```

### Test 4: Frontend Chat

1. Navigate to `http://localhost:5173/chat`
2. Select season (Kharif, Rabi, Zaid)
3. Type a query:
   - "What crops should I grow?"
   - "How to prevent diseases?"
   - "What's the soil pH needed?"
   - "Best market prices for crops?"

---

## 📊 API Response Examples

### RAG Query Response

```json
{
  "query": "What crops for Kharif?",
  "intent": "crop_recommendation",
  "confidence": 0.95,
  "response_text": "Based on your requirements, I recommend: Rice, Maize, Cotton. These crops are well-suited to Kharif season.",
  "data": {
    "recommendations": [
      {
        "crop_name": "Rice",
        "scientific_name": "Oryza sativa",
        "season": "Kharif",
        "crop_type": "Cereal",
        "min_temp": 20,
        "max_temp": 35,
        "water_requirement": 9.5,
        "npk": "120-60-60",
        "growth_days": 120
      },
      // ... more crops
    ]
  },
  "timestamp": "2024-01-05T10:30:00.000Z"
}
```

### Crop Recommendation Response

```json
{
  "crop_name": "Test",
  "recommended_crop": "Rice",
  "confidence": 0.87,
  "model_type": "crop_recommendation",
  "timestamp": "2024-01-05T10:30:00.000Z"
}
```

---

## 🔧 Configuration & Customization

### Changing Model Hyperparameters

Edit `backend/ml/train_models.py`:

```python
# Crop Recommendation Model
model = RandomForestClassifier(
    n_estimators=200,      # Increase for more trees
    max_depth=20,          # Limit tree depth
    random_state=42,
    n_jobs=-1
)
```

### Adding Custom Intent Categories

Edit `backend/ml/rag_pipeline.py`:

```python
class IntentClassifier:
    INTENTS = {
        'weather': [...],
        'disease': [...],
        'soil': [...],
        'crop_recommendation': [...],
        'pricing': [...],
        # Add your custom intent:
        'equipment': ['machine', 'tool', 'equipment', 'tractor'],
    }
```

### Modifying RAG Response Generation

Edit `backend/ml/rag_pipeline.py` - `_generate_response()` method:

```python
def _generate_response(self, intent: str, data: Dict) -> str:
    if intent == 'custom_intent':
        return f"Custom response for {intent}"
```

---

## ⚠️ Important Notes

### Model Performance
- **Crop Recommendation**: 0% accuracy (due to 96 unique classes with 1 sample each)
  - Solution: Implement similarity-based matching instead of classification
  - Or: Add more training data per crop

- **Growth Duration**: R² = 0.75 (good)
- **Season Classification**: Accuracy = 75% (good)

### Improving Models
1. **Add More Training Data** - Collect real-world data
2. **Feature Engineering** - Create domain-specific features
3. **Ensemble Methods** - Combine multiple models
4. **Hyperparameter Tuning** - Use GridSearchCV

### Scaling Considerations
- Models load in-memory on startup (~500MB)
- RAG pipeline queries CSV data (cached in-memory)
- For large datasets, use database with indexing

---

## 📝 Example: Complete User Flow

1. **User types**: "What should I grow in Kharif with clay loam soil?"

2. **Intent Classification**:
   - Intent: `crop_recommendation`
   - Confidence: 0.92

3. **Retrieval**:
   - Search by: season=Kharif, soil_type=Clay_Loam
   - Returns: [Rice, Wheat, Cotton, ...]

4. **Generation**:
   - Response: "For Kharif season with clay loam soil, I recommend Rice, Cotton, and Wheat..."

5. **Frontend**:
   - Displays response with crop cards
   - Shows intent badge
   - Shows confidence score

---

## 🔗 Next Steps

### Production Deployment
1. Use production WSGI server (Gunicorn)
2. Cache RAG pipeline outputs in Redis
3. Monitor model performance
4. Set up CI/CD for model retraining

### Model Improvement
1. Collect more crop data
2. Implement active learning
3. Add A/B testing for responses
4. Track user satisfaction

### Feature Additions
1. Multi-language support
2. User feedback loop
3. Personalized recommendations
4. Historical data tracking

---

## 📚 Resources

- Model training script: `backend/ml/train_models.py`
- RAG implementation: `backend/ml/rag_pipeline.py`
- Inference utilities: `backend/ml/inference.py`
- API endpoints: `backend/api/routes/ml_predictions.py`
- Frontend component: `frontend/src/components/AgriSenseRAGChat.tsx`

---

## ✅ Checklist

- [ ] Install dependencies
- [ ] Mount ML routes in FastAPI app
- [ ] Initialize RAG pipeline on startup
- [ ] Add RAG chat component to frontend
- [ ] Configure API base URL
- [ ] Test health endpoint
- [ ] Test RAG queries
- [ ] Test crop predictions
- [ ] Deploy to production

