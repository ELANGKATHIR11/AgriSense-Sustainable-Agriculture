"""
FastAPI ML Service Wrapper
Provides high-performance REST API for all ML models
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np

# Import native ML services
try:
    from native_agricultural_advisor import get_agricultural_advice

    ADVISOR_AVAILABLE = True
except ImportError:
    ADVISOR_AVAILABLE = False
    logging.warning("Native agricultural advisor not available")

import os

# Import other ML models
import sys

sys.path.append(os.path.dirname(__file__))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AgriSense ML API",
    description="High-performance ML inference API for agricultural predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AgriculturalQuery(BaseModel):
    query: str = Field(
        ..., min_length=3, max_length=1000, description="Agricultural question"
    )
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class CropRecommendationInput(BaseModel):
    N: float = Field(..., ge=0, le=300, description="Nitrogen content")
    P: float = Field(..., ge=0, le=300, description="Phosphorus content")
    K: float = Field(..., ge=0, le=300, description="Potassium content")
    temperature: float = Field(..., ge=-10, le=60, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(..., ge=0, le=1000, description="Rainfall in mm")


class YieldPredictionInput(BaseModel):
    Area: float = Field(..., gt=0, description="Area in hectares")
    Item: str = Field(..., description="Crop name")
    Year: int = Field(..., ge=2000, le=2100, description="Year")
    average_rain_fall_mm_per_year: Optional[float] = None
    pesticides_tonnes: Optional[float] = None
    avg_temp: Optional[float] = None


# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Duration: {duration:.3f}s"
    )

    return response


# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "AgriSense ML API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": [
            "/docs - API Documentation",
            "/health - Health check",
            "/agricultural-advice - Get farming advice",
            "/crop-recommendation - Predict best crop",
            "/yield-prediction - Predict crop yield",
        ],
    }


# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "agricultural_advisor": ADVISOR_AVAILABLE,
            "crop_recommendation": True,  # Add actual checks
            "yield_prediction": True,
        },
    }


# Agricultural Advice Endpoint
@app.post("/agricultural-advice")
async def get_advice(query: AgriculturalQuery):
    """
    Get agricultural advice from native Phi-2 model or rule-based fallback
    """
    if not ADVISOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Agricultural advisor service is not available",
        )

    try:
        start_time = time.time()

        result = get_agricultural_advice(query.query, query.context)

        duration = time.time() - start_time

        return {
            "success": True,
            "data": result,
            "metadata": {
                "duration_ms": int(duration * 1000),
                "timestamp": datetime.now().isoformat(),
            },
        }
    except Exception as e:
        logger.error(f"Agricultural advice error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Crop Recommendation Endpoint
@app.post("/crop-recommendation")
async def crop_recommendation(input_data: CropRecommendationInput):
    """
    Recommend best crop based on soil and climate parameters
    """
    try:
        import pickle

        # Load model
        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "enhanced_crop_recommendation_model.pkl",
        )
        scaler_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "enhanced_crop_recommendation_scaler.pkl",
        )

        if not os.path.exists(model_path):
            raise HTTPException(status_code=503, detail="Model not found")

        model = pickle.load(open(model_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))

        # Prepare input
        features = np.array(
            [
                [
                    input_data.N,
                    input_data.P,
                    input_data.K,
                    input_data.temperature,
                    input_data.humidity,
                    input_data.ph,
                    input_data.rainfall,
                ]
            ]
        )

        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Get top 3 recommendations
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        recommendations = [
            {
                "crop": model.classes_[idx],
                "confidence": float(probabilities[idx]),
            }
            for idx in top_3_indices
        ]

        return {
            "success": True,
            "data": {
                "recommended_crop": prediction,
                "confidence": float(probabilities.max()),
                "alternatives": recommendations[1:],
                "all_scores": recommendations,
            },
            "metadata": {
                "model": "RandomForestClassifier",
                "version": "enhanced_v1",
                "timestamp": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"Crop recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Yield Prediction Endpoint
@app.post("/yield-prediction")
async def yield_prediction(input_data: YieldPredictionInput):
    """
    Predict crop yield based on historical and environmental data
    """
    try:
        import pickle

        import pandas as pd

        # Load model
        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "enhanced_yield_prediction_model.pkl",
        )
        scaler_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "enhanced_yield_prediction_scaler.pkl",
        )

        if not os.path.exists(model_path):
            raise HTTPException(status_code=503, detail="Model not found")

        model = pickle.load(open(model_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))

        # Prepare input (simplified - adjust based on actual model features)
        features_dict = {
            "Area": input_data.Area,
            "Year": input_data.Year,
            "average_rain_fall_mm_per_year": input_data.average_rain_fall_mm_per_year
            or 0,
            "pesticides_tonnes": input_data.pesticides_tonnes or 0,
            "avg_temp": input_data.avg_temp or 25,
        }

        # Create dataframe (model might need specific feature engineering)
        features = pd.DataFrame([features_dict])

        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        return {
            "success": True,
            "data": {
                "predicted_yield": float(prediction),
                "unit": "hg/ha (hectograms per hectare)",
                "crop": input_data.Item,
                "area": input_data.Area,
                "year": input_data.Year,
            },
            "metadata": {
                "model": "GradientBoostingRegressor",
                "version": "enhanced_v1",
                "timestamp": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"Yield prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn fastapi_service:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
