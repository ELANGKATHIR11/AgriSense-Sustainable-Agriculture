"""
FastAPI endpoints for ML predictions and RAG queries
Integrates trained models with the AgriSense backend
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
import logging

from ml.inference import get_inference_engine, ModelInference
from ml.rag_pipeline import RAGPipeline, initialize_rag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["machine-learning"])

# Global RAG pipeline instance
_rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or initialize RAG pipeline"""
    global _rag_pipeline
    if _rag_pipeline is None:
        try:
            _rag_pipeline = initialize_rag_pipeline()
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    return _rag_pipeline


# ==================== Request/Response Models ====================

class PredictionRequest(BaseModel):
    """Request for ML predictions"""
    crop_name: str = Field(..., description="Name of the crop")
    features: List[float] = Field(..., description="Feature vector for prediction")
    model_type: str = Field("crop_recommendation", description="Type of prediction")


class RAGQueryRequest(BaseModel):
    """Request for RAG-based queries"""
    query: str = Field(..., description="User query/question")
    season: Optional[str] = Field(None, description="Current season")
    crop_type: Optional[str] = Field(None, description="Crop type filter")
    location: Optional[str] = Field(None, description="User location")


class CropRecommendationRequest(BaseModel):
    """Request for crop recommendations"""
    min_temperature: float
    max_temperature: float
    pH: float
    rainfall: float
    water_availability: float
    soil_type: Optional[str] = None
    season: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response for predictions"""
    crop_name: str
    model_type: str
    prediction: Any
    confidence: Optional[float] = None
    timestamp: str


class RAGResponse(BaseModel):
    """Response for RAG queries"""
    query: str
    intent: str
    confidence: float
    response_text: str
    data: Dict[str, Any]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Information about available models"""
    models: List[str]
    status: str
    metrics: Dict[str, Any]
    timestamp: str


# ==================== Model Health & Info ====================

@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Check if ML models are loaded and ready"""
    try:
        engine = get_inference_engine()
        info = engine.get_model_info()
        return {
            "status": info['status'],
            "models_count": len(info['models_loaded']),
            "models": ', '.join(info['models_loaded'])
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="ML service unavailable")


@router.get("/models/info", response_model=ModelInfoResponse)
async def get_models_info():
    """Get information about all trained models"""
    try:
        engine = get_inference_engine()
        info = engine.get_model_info()
        return ModelInfoResponse(
            models=info['models_loaded'],
            status=info['status'],
            metrics=info['metrics'],
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Individual Predictions ====================

@router.post("/predict/crop-recommendation")
async def predict_crop_recommendation(request: PredictionRequest):
    """
    Predict recommended crop
    
    Features (19):
    - Temperature range, pH range, soil properties
    - Water and nutrient requirements
    """
    try:
        engine = get_inference_engine()
        features = np.array(request.features)
        
        if len(features) != 19:
            raise ValueError(f"Expected 19 features, got {len(features)}")
        
        crop, confidence = engine.predict_crop_recommendation(features)
        
        return {
            "crop_name": request.crop_name,
            "recommended_crop": crop,
            "confidence": confidence,
            "model_type": "crop_recommendation",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/crop-type")
async def predict_crop_type(request: PredictionRequest):
    """
    Predict crop type (Cash, Cereal, Fruit, etc.)
    
    Features (26): Extended feature set with engineered features
    """
    try:
        engine = get_inference_engine()
        features = np.array(request.features)
        
        if len(features) != 26:
            raise ValueError(f"Expected 26 features, got {len(features)}")
        
        crop_type, probabilities = engine.predict_crop_type(features)
        
        return {
            "crop_name": request.crop_name,
            "predicted_crop_type": crop_type,
            "probabilities": probabilities,
            "model_type": "crop_type_classification",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/growth-duration")
async def predict_growth_duration(request: PredictionRequest):
    """
    Predict growth duration in days
    
    Features (23): Selected features for regression
    """
    try:
        engine = get_inference_engine()
        features = np.array(request.features)
        
        if len(features) != 23:
            raise ValueError(f"Expected 23 features, got {len(features)}")
        
        days, metrics = engine.predict_growth_duration(features)
        
        return {
            "crop_name": request.crop_name,
            "predicted_days": round(days, 0),
            "metrics": metrics,
            "model_type": "growth_duration",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/water-requirement")
async def predict_water_requirement(request: PredictionRequest):
    """
    Predict daily water requirement in mm
    
    Features (19): Soil, climate, and crop-specific features
    """
    try:
        engine = get_inference_engine()
        features = np.array(request.features)
        
        if len(features) != 19:
            raise ValueError(f"Expected 19 features, got {len(features)}")
        
        water, metrics = engine.predict_water_requirement(features)
        
        return {
            "crop_name": request.crop_name,
            "predicted_water_requirement_mm_day": round(water, 2),
            "metrics": metrics,
            "model_type": "water_requirement",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/season")
async def predict_season(request: PredictionRequest):
    """
    Predict suitable season for crop (Kharif, Rabi, Zaid, etc.)
    
    Features (20): Climate and seasonal parameters
    """
    try:
        engine = get_inference_engine()
        features = np.array(request.features)
        
        if len(features) != 20:
            raise ValueError(f"Expected 20 features, got {len(features)}")
        
        season, probabilities = engine.predict_season(features)
        
        return {
            "crop_name": request.crop_name,
            "predicted_season": season,
            "probabilities": probabilities,
            "model_type": "season_classification",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Batch Predictions ====================

@router.post("/predict/batch")
async def batch_predictions(request: PredictionRequest):
    """
    Get all predictions for a crop at once
    Returns: crop type, growth duration, water requirement, season
    """
    try:
        engine = get_inference_engine()
        
        # Note: In production, would prepare proper feature dictionaries
        results = {
            "crop_name": request.crop_name,
            "predictions": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== RAG Pipeline Endpoints ====================

@router.post("/rag/query", response_model=RAGResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Process user query through RAG pipeline
    
    Combines:
    1. Intent Classification (SVM)
    2. Retrieval (Cosine Similarity)
    3. Generation (Natural Language Response)
    """
    try:
        pipeline = get_rag_pipeline()
        
        context = {
            'season': request.season,
            'crop_type': request.crop_type,
            'location': request.location
        }
        
        result = pipeline.process_query(request.query, context)
        
        return RAGResponse(
            query=request.query,
            intent=result['intent'],
            confidence=result['confidence'],
            response_text=result['response_text'],
            data={
                'recommendations': result.get('recommendations', []),
                'weather_info': result.get('weather_info', {}),
                'disease_info': result.get('disease_info', {}),
                'soil_info': result.get('soil_info', {}),
                'pricing_info': result.get('pricing_info', {})
            },
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/rag/intents")
async def get_intents():
    """Get available intent categories"""
    try:
        pipeline = get_rag_pipeline()
        return {
            "intents": pipeline.intent_classifier.INTENTS,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/classify-intent")
async def classify_intent(query: str = Query(..., description="User query")):
    """Classify the intent of a user query"""
    try:
        pipeline = get_rag_pipeline()
        intent, confidence = pipeline.intent_classifier.classify(query)
        
        return {
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "keywords": pipeline.intent_classifier.get_keywords_for_intent(intent),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Crop Search & Recommendations ====================

@router.get("/crops/search")
async def search_crops(
    season: Optional[str] = Query(None),
    crop_type: Optional[str] = Query(None),
    min_temp: Optional[float] = Query(None),
    max_temp: Optional[float] = Query(None)
):
    """Search crops by criteria"""
    try:
        pipeline = get_rag_pipeline()
        
        temp_range = None
        if min_temp is not None and max_temp is not None:
            temp_range = (min_temp, max_temp)
        
        results = pipeline.retriever.search_by_criteria(
            season=season,
            crop_type=crop_type,
            temp_range=temp_range
        )
        
        return {
            "total": len(results),
            "crops": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Crop search error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/crops/recommendations")
async def get_crop_recommendations(
    season: str = Query(...),
    limit: int = Query(5, ge=1, le=20)
):
    """Get crop recommendations for a season"""
    try:
        pipeline = get_rag_pipeline()
        crops = pipeline.retriever.search_by_criteria(season=season)
        
        return {
            "season": season,
            "total": len(crops),
            "recommendations": crops[:limit],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Testing Endpoints ====================

@router.post("/test/predict")
async def test_prediction():
    """Test endpoint with sample data"""
    try:
        # Sample feature vector
        sample_features = np.random.randn(19)
        request = PredictionRequest(
            crop_name="Test Crop",
            features=sample_features.tolist(),
            model_type="crop_recommendation"
        )
        return await predict_crop_recommendation(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/rag")
async def test_rag():
    """Test RAG pipeline"""
    try:
        request = RAGQueryRequest(
            query="What crops should I plant in Kharif season?",
            season="Kharif"
        )
        return await rag_query(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount router
def mount_ml_routes(app):
    """Mount ML routes to FastAPI app"""
    app.include_router(router)
