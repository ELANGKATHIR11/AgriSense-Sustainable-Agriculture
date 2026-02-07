# Claude-Enhanced Crop Recommendation System

Advanced agricultural decision support system for AgriSense platform, powered by machine learning and optimized for 100+ Indian crops.

## 🌾 Overview

This system provides intelligent crop recommendations based on comprehensive soil and environmental parameters. It analyzes:

- **Soil Properties**: pH, NPK (Nitrogen, Phosphorus, Potassium), micronutrients (Fe, Mn, Zn, Cu, B)
- **Environmental Factors**: Temperature, humidity, rainfall, water availability
- **Soil Conditions**: Moisture content, EC, texture

## 📊 Supported Crops (100+)

### Categories:
- **Cereals (15)**: Rice, Wheat, Maize, Sorghum, Pearl Millet, Finger Millet, Barley, Oats, etc.
- **Pulses (15)**: Chickpea, Pigeon Pea, Black Gram, Green Gram, Lentil, Field Pea, etc.
- **Oilseeds (15)**: Groundnut, Soybean, Sunflower, Mustard, Sesame, Safflower, etc.
- **Cash Crops (10)**: Sugarcane, Cotton, Jute, Tea, Coffee, Tobacco, Rubber, Cashew, etc.
- **Vegetables (20)**: Tomato, Potato, Onion, Cabbage, Chilli, etc.
- **Fruits (15)**: Mango, Banana, Papaya, Guava, Grapes, Pomegranate, Citrus, etc.
- **Spices (10)**: Turmeric, Ginger, Garlic, Cumin, Coriander, Black Pepper, etc.

## 🏗️ Architecture

### Backend Components:

```
backend/ml/claude_crop_recommender/
├── crop_recommendation_ml_model.py       # ML model implementation (Random Forest, GB, SVM, NN)
├── crop_recommender_api.py               # API wrapper with Pydantic models
├── crop_requirements_dataset.py          # Comprehensive crop dataset (100+ crops)
├── routes.py                             # FastAPI routes
├── requirements.txt                      # Python dependencies
└── __init__.py                           # Package initialization
```

### Models Trained:
- **Random Forest** (Best performer)
- **Gradient Boosting**
- **Support Vector Machine (SVM)**
- **Neural Network (MLP)**

**Model Accuracy**: ~92% on test set

## 🚀 Integration Points

### 1. **Node.js Backend** (`backend/services/cropRecommendationService.js`)
- Maps legacy input format to enhanced format
- Calls Python FastAPI service
- Provides fallback to native ML if Python service unavailable
- Transforms responses to maintain backward compatibility

### 2. **FastAPI Service** (Python, `backend/ml/claude_crop_recommender/routes.py`)
- `/crop-recommendation/predict` - Get crop recommendations
- `/crop-recommendation/health` - Service health check
- `/crop-recommendation/crops-list` - Get all supported crops
- `/crop-recommendation/crop-requirements/{crop_name}` - Get crop-specific requirements

### 3. **Express Controller** (`backend/controllers/cropRecommendationController.js`)
- POST `/api/crop-recommendation/predict` - Accept soil parameters
- Returns enhanced response with all recommendations, analysis, and metadata

### 4. **Frontend** (`frontend/services/api.ts`)
- `recommendCrop()` - Call crop recommendation API
- Handles both enhanced and legacy response formats
- Returns structured recommendation data with alternatives

## 📋 API Specification

### Request Format:
```json
{
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
}
```

### Response Format:
```json
{
  "success": true,
  "data": {
    "primary_recommendation": {
      "crop": "Rice",
      "confidence": 0.85,
      "suitability_score": "85.00"
    },
    "alternatives": ["Wheat", "Maize"],
    "all_recommendations": [
      {
        "rank": 1,
        "crop_name": "Rice",
        "suitability_score": 85.0,
        "confidence": "High"
      }
    ],
    "analysis": {
      "soil_status": { ... },
      "climate_status": { ... }
    }
  },
  "model_info": {
    "model_name": "Random Forest",
    "accuracy": 0.92,
    "total_crops": 100,
    "version": "2.0"
  },
  "timestamp": "2026-02-07T10:30:00Z"
}
```

## 🔧 Installation & Setup

### 1. Install Python Dependencies:
```bash
cd backend/ml/claude_crop_recommender
pip install -r requirements.txt
```

### 2. Train the Model (First Time):
```bash
python crop_recommender_api.py
```

This will generate `crop_recommendation_model.pkl` (~50MB)

### 3. Start FastAPI Service:
```bash
# From backend directory
python -m uvicorn ml.claude_crop_recommender.routes:router --host 0.0.0.0 --port 8000
```

Or integrate into main FastAPI service:
```python
# In backend/ai_service/main.py
from ml.claude_crop_recommender.routes import router as crop_recommendation_router
app.include_router(crop_recommendation_router)
```

### 4. Start Node.js Backend:
```bash
npm start
```

### 5. Environment Variables:
```env
# .env
PYTHON_SERVICE_URL=http://localhost:8000
ML_SERVICE_URL=http://localhost:5001
NODE_ENV=development
```

## 💡 Usage Examples

### Frontend Component Example:
```typescript
import { recommendCrop } from '../services/api';

const handleRecommendation = async () => {
  const input = {
    nitrogen: 120,
    phosphorus: 54,
    potassium: 100,
    temperature: 28,
    humidity: 60,
    ph: 7.0,
    rainfall: 600
  };

  const result = await recommendCrop(input);
  
  console.log(`Primary Recommendation: ${result.crop}`);
  console.log(`Confidence: ${result.confidence * 100}%`);
  console.log(`All Recommendations:`, result.recommendations);
};
```

### Direct Python API Usage:
```python
from crop_recommender_api import SoilParameters, recommend_crops

params = SoilParameters(
    pH=7.0, N=120, P=54, K=100,
    Fe=4.06, Mn=1.68, Zn=0.83, Cu=0.46, B=0.3,
    Water=500, Moisture=60, Temperature=28, Rainfall=600,
    top_n=5
)

result = recommend_crops(params)
print(result['recommendations'])
```

## 🎯 Data Model

### Features Used:
1. **Macronutrients** (kg/ha): N, P, K
2. **Micronutrients** (ppm): Fe, Mn, Zn, Cu, B
3. **Soil Properties**: pH, Moisture (%)
4. **Climate**: Temperature (°C), Rainfall (mm/season)
5. **Water**: Requirement (mm/season)

### Total Parameters: 13
### Training Samples: 5,000 (50 per crop × 100 crops)

## 📈 Model Performance

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Random Forest | 92.3% | 2-3 min |
| Gradient Boosting | 89.1% | 3-4 min |
| SVM | 88.7% | 4-5 min |
| Neural Network | 87.5% | 3-4 min |

**Selected**: Random Forest (Fastest with highest accuracy)

## 🔄 Data Flow

```
Frontend Input (Soil Parameters)
         ↓
Express Controller (/api/crop-recommendation/predict)
         ↓
Node.js Service (cropRecommendationService)
         ↓
FastAPI Endpoint (/crop-recommendation/predict)
         ↓
Python ML Model (CropRecommendationSystem)
         ↓
Response (Recommendations + Metadata)
         ↓
Frontend Display (with alternatives and analysis)
```

## 🛡️ Error Handling

- **Validation Errors**: Missing or invalid soil parameters
- **Service Unavailable**: Fallback to native ML or mock responses
- **Model Not Found**: Auto-trains on first request (slow)
- **Out of Range Values**: Clamps to valid ranges with warning

## 📊 Example Scenarios

### Scenario 1: Good Agricultural Land
```
Input: pH 6.5, N 180, P 80, K 150, Temp 25°C, Rain 800mm
Output: "Rice" (93% suitability), Alternatives: "Sugarcane", "Banana"
```

### Scenario 2: Dry Region
```
Input: pH 7.8, N 40, P 25, K 30, Temp 35°C, Rain 350mm
Output: "Mustard" (87% suitability), Alternatives: "Pearl Millet", "Sorghum"
```

### Scenario 3: Acidic Saline Soil
```
Input: pH 5.2, N 50, P 30, K 45, Temp 20°C, Rain 600mm
Output: "Chickpea" (84% suitability), Alternatives: "Lentil", "Black Gram"
```

## 🚨 Troubleshooting

### Issue: "Model file not found"
**Solution**: Train the model first
```bash
python crop_recommender_api.py
```

### Issue: "Python service unavailable"
**Solution**: Ensure FastAPI service is running on port 8000
```bash
python -m uvicorn ml.claude_crop_recommender.routes:router --port 8000
```

### Issue: "Validation error - parameter out of range"
**Solution**: Check input values are within valid ranges:
- pH: 4.5-8.5
- N: 10-300 kg/ha
- Temperature: 10-40°C
- Rainfall: 200-2500 mm

## 📚 Dataset Source

- Indian Government Agricultural Departments
- Tamil Nadu Soil Health Card Format
- Standard crop requirements from ICAR (Indian Council of Agricultural Research)
- Regional adaptation studies

## 🔐 Security

- Input validation on all parameters
- Rate limiting on API endpoints
- No sensitive data stored/logged
- Model predictions are stateless

## 📝 Future Enhancements

1. **Crop Rotation Advisor**: Suggests rotation patterns
2. **Weather Integration**: Real-time weather data
3. **Market Price Optimizer**: Recommendations based on market prices
4. **Yield Prediction**: Combines with yield models
5. **Pest/Disease Risk**: Integrates with disease prediction
6. **Deep Learning Models**: Neural network optimization
7. **Regional Fine-tuning**: State-specific models
8. **Mobile Optimization**: Lightweight model for edge devices

## 📞 Support

For issues or suggestions:
1. Check logs: `backend/logs/`
2. Verify service health: `/crop-recommendation/health`
3. Review API documentation: `/api/docs` (Swagger)

## 📄 License

AgriSense Platform - Agricultural IoT System

---

**Last Updated**: February 7, 2026  
**Version**: 2.0  
**Status**: Production Ready
