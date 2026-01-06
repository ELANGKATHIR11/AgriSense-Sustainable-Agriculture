"""
ML RAG Pipeline Integration Routes
Hybrid RAG system with intent classification, semantic retrieval, and response generation
"""

import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================

class RAGQueryRequest(BaseModel):
    """RAG query request model"""
    query: str = Field(..., description="User query about crops")
    season: Optional[str] = Field(None, description="Current/planned season (e.g., kharif, rabi)")
    crop_type: Optional[str] = Field(None, description="Crop type context")
    location: Optional[str] = Field(None, description="Location for context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What crops should I grow in Kharif?",
                "season": "kharif",
                "crop_type": "cereals",
                "location": "Karnataka"
            }
        }


class PredictionRequest(BaseModel):
    """Prediction request model"""
    crop_name: str = Field(..., description="Name of crop to analyze")
    features: List[float] = Field(..., description="Feature vector for prediction")
    model_type: Optional[str] = Field(None, description="Specific model to use")
    
    class Config:
        json_schema_extra = {
            "example": {
                "crop_name": "rice",
                "features": [25.5, 60.0, 800.0, 6.5, 50, 15, 100, 75, 20, 5, 40, 30, 1, 0, 0, 1, 0, 0, 0]
            }
        }


class RAGResponse(BaseModel):
    """RAG response model"""
    query: str
    intent: str = Field(..., description="Detected intent (weather/disease/soil/crop_recommendation/pricing)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    response_text: str = Field(..., description="Generated response")
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PredictionResponse(BaseModel):
    """Prediction response model"""
    crop_name: str
    model_type: str
    prediction: Any
    confidence: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    """Model information response"""
    status: str
    models_loaded: int
    model_names: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ml_engine_ready: bool
    rag_pipeline_ready: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Router Definition
# ============================================================================

router = APIRouter(prefix="/v1/ml", tags=["ML & RAG"])


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check health of ML system
    
    Returns:
        - status: "ok" if system is ready
        - ml_engine_ready: Whether inference engine is loaded
        - rag_pipeline_ready: Whether RAG pipeline is initialized
    """
    try:
        from ml.inference import get_inference_engine
        from ml.rag_pipeline import initialize_rag_pipeline
        
        engine = get_inference_engine()
        pipeline = initialize_rag_pipeline()
        
        return HealthResponse(
            status="ok",
            ml_engine_ready=engine is not None,
            rag_pipeline_ready=pipeline is not None
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            ml_engine_ready=False,
            rag_pipeline_ready=False
        )


@router.get("/models/info", response_model=ModelInfoResponse)
async def get_models_info():
    """
    Get information about loaded ML models
    
    Returns:
        - Number of models loaded
        - Model names
        - Performance metrics
    """
    try:
        from ml.inference import get_inference_engine
        
        engine = get_inference_engine()
        info = engine.get_model_info() if engine else {}
        
        return ModelInfoResponse(
            status="ok",
            models_loaded=len(info.get("models", [])),
            model_names=info.get("models", []),
            metrics=info.get("metrics", {})
        )
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RAG Pipeline Endpoints
# ============================================================================

@router.post("/rag/query", response_model=RAGResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Process user query through RAG pipeline
    
    Flow:
    1. Intent Classification - Determines query type (weather/disease/soil/crop/pricing)
    2. Semantic Retrieval - Finds relevant crops and data
    3. Response Generation - Formats natural language response
    
    Args:
        query: User question
        season: Optional season context
        crop_type: Optional crop filter
        location: Optional location context
        
    Returns:
        - intent: Detected intent type
        - confidence: Confidence in intent classification
        - response_text: Generated response
        - data: Retrieved structured data
    """
    try:
        from ml.rag_pipeline import initialize_rag_pipeline
        
        pipeline = initialize_rag_pipeline()
        
        context = {}
        if request.season:
            context["season"] = request.season
        if request.crop_type:
            context["crop_type"] = request.crop_type
        if request.location:
            context["location"] = request.location
        
        result = pipeline.process_query(request.query, context)
        
        return RAGResponse(
            query=request.query,
            intent=result.get("intent", "unknown"),
            confidence=result.get("confidence", 0.0),
            response_text=result.get("response_text", ""),
            data=result.get("data", {})
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.post("/rag/classify-intent")
async def classify_intent(query: str):
    """
    Classify query intent without full RAG processing
    
    Returns:
        - intent: One of [weather, disease, soil, crop_recommendation, pricing]
        - confidence: Confidence score
    """
    try:
        from ml.rag_pipeline import initialize_rag_pipeline
        
        pipeline = initialize_rag_pipeline()
        intent, confidence = pipeline.classifier.classify(query)
        
        return {
            "intent": intent,
            "confidence": float(confidence),
            "intents": ["weather", "disease", "soil", "crop_recommendation", "pricing"]
        }
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Prediction Endpoints
# ============================================================================

@router.post("/predict/crop-recommendation", response_model=PredictionResponse)
async def predict_crop_recommendation(request: PredictionRequest):
    """
    Predict crop recommendations based on features
    
    Features (19 required):
        - temperature, humidity, rainfall, pH, nitrogen, phosphorus, potassium,
        - organic_matter, soil_type_encoded, season_encoded, etc.
    """
    try:
        from ml.inference import make_prediction
        
        result = make_prediction(
            "crop_recommendation",
            {request.crop_name: request.features}
        )
        
        return PredictionResponse(
            crop_name=request.crop_name,
            model_type="crop_recommendation",
            prediction=result.get("prediction"),
            confidence=result.get("confidence"),
            metrics=result.get("metrics")
        )
    except Exception as e:
        logger.error(f"Crop recommendation prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/crop-type", response_model=PredictionResponse)
async def predict_crop_type(request: PredictionRequest):
    """
    Classify crop type from features
    
    Returns:
        - prediction: Predicted crop type
        - confidence: Confidence score
    """
    try:
        from ml.inference import make_prediction
        
        result = make_prediction(
            "crop_type",
            {request.crop_name: request.features}
        )
        
        return PredictionResponse(
            crop_name=request.crop_name,
            model_type="crop_type",
            prediction=result.get("prediction"),
            confidence=result.get("confidence"),
            metrics=result.get("metrics")
        )
    except Exception as e:
        logger.error(f"Crop type prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/growth-duration", response_model=PredictionResponse)
async def predict_growth_duration(request: PredictionRequest):
    """
    Predict growth duration in days
    
    Returns:
        - prediction: Expected growth duration (days)
        - metrics: R² score and other metrics
    """
    try:
        from ml.inference import make_prediction
        
        result = make_prediction(
            "growth_duration",
            {request.crop_name: request.features}
        )
        
        return PredictionResponse(
            crop_name=request.crop_name,
            model_type="growth_duration",
            prediction=result.get("prediction"),
            metrics=result.get("metrics")
        )
    except Exception as e:
        logger.error(f"Growth duration prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/water-requirement", response_model=PredictionResponse)
async def predict_water_requirement(request: PredictionRequest):
    """
    Predict water requirement in mm/day
    
    Returns:
        - prediction: Water needed (mm/day)
        - metrics: R² score and other metrics
    """
    try:
        from ml.inference import make_prediction
        
        result = make_prediction(
            "water_requirement",
            {request.crop_name: request.features}
        )
        
        return PredictionResponse(
            crop_name=request.crop_name,
            model_type="water_requirement",
            prediction=result.get("prediction"),
            metrics=result.get("metrics")
        )
    except Exception as e:
        logger.error(f"Water requirement prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/season", response_model=PredictionResponse)
async def predict_season(request: PredictionRequest):
    """
    Predict season suitability
    
    Returns:
        - prediction: Predicted season
        - confidence: Confidence score
    """
    try:
        from ml.inference import make_prediction
        
        result = make_prediction(
            "season",
            {request.crop_name: request.features}
        )
        
        return PredictionResponse(
            crop_name=request.crop_name,
            model_type="season",
            prediction=result.get("prediction"),
            confidence=result.get("confidence"),
            metrics=result.get("metrics")
        )
    except Exception as e:
        logger.error(f"Season prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch")
async def batch_predict(requests: List[PredictionRequest]):
    """
    Make multiple predictions in batch
    
    Returns:
        - results: List of prediction results
        - count: Number of predictions
        - duration_ms: Processing time
    """
    try:
        from ml.inference import make_prediction
        import time
        
        start = time.time()
        results = []
        
        for req in requests:
            try:
                for model_type in ["crop_type", "growth_duration", "water_requirement", "season"]:
                    result = make_prediction(model_type, {req.crop_name: req.features})
                    results.append({
                        "crop": req.crop_name,
                        "model": model_type,
                        "prediction": result.get("prediction"),
                        "confidence": result.get("confidence")
                    })
            except Exception as e:
                logger.warning(f"Batch prediction failed for {req.crop_name}: {e}")
        
        duration_ms = (time.time() - start) * 1000
        
        return {
            "status": "ok",
            "count": len(results),
            "duration_ms": duration_ms,
            "results": results
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Search & Recommendation Endpoints
# ============================================================================

@router.get("/crops/search")
async def search_crops(query: str, season: Optional[str] = None, top_k: int = 5):
    """
    Search for crops matching criteria
    
    Args:
        query: Search query
        season: Optional season filter
        top_k: Number of results to return
        
    Returns:
        - crops: List of matching crops
        - count: Number of results
    """
    try:
        from ml.rag_pipeline import initialize_rag_pipeline
        
        pipeline = initialize_rag_pipeline()
        results = pipeline.retriever.search(query, top_k=top_k)
        
        return {
            "status": "ok",
            "query": query,
            "count": len(results),
            "crops": results
        }
    except Exception as e:
        logger.error(f"Crop search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/crops/recommendations")
async def get_crop_recommendations(season: str, soil_type: Optional[str] = None):
    """
    Get recommended crops for given season
    
    Args:
        season: Season (kharif/rabi/summer)
        soil_type: Optional soil type filter
        
    Returns:
        - recommendations: List of recommended crops
        - season: Requested season
        - soil_type: Soil type used
    """
    try:
        from ml.rag_pipeline import initialize_rag_pipeline
        
        pipeline = initialize_rag_pipeline()
        
        context = {"season": season}
        if soil_type:
            context["soil_type"] = soil_type
        
        # Query RAG pipeline for recommendations
        query = f"Which crops should I grow in {season}?"
        result = pipeline.process_query(query, context)
        
        return {
            "status": "ok",
            "season": season,
            "soil_type": soil_type,
            "recommendations": result.get("data", {}).get("recommendations", []),
            "response_text": result.get("response_text", "")
        }
    except Exception as e:
        logger.error(f"Crop recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Test Endpoints
# ============================================================================

@router.get("/test/predict")
async def test_predict():
    """Test endpoint for prediction"""
    try:
        from ml.inference import make_prediction
        
        # Test with example features for rice
        test_features = [25.5, 60.0, 800.0, 6.5, 50, 15, 100, 75, 20, 5, 40, 30, 1, 0, 0, 1, 0, 0, 0]
        result = make_prediction("crop_type", {"rice": test_features})
        
        return {
            "status": "ok",
            "test": "crop_type_prediction",
            "crop": "rice",
            "result": result
        }
    except Exception as e:
        logger.error(f"Test prediction failed: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/test/rag")
async def test_rag():
    """Test endpoint for RAG pipeline"""
    try:
        from ml.rag_pipeline import initialize_rag_pipeline
        
        pipeline = initialize_rag_pipeline()
        result = pipeline.process_query(
            "What crops should I grow in Kharif?",
            {"season": "kharif"}
        )
        
        return {
            "status": "ok",
            "test": "rag_query",
            "query": "What crops should I grow in Kharif?",
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "response": result.get("response_text")[:100] + "..."  # First 100 chars
        }
    except Exception as e:
        logger.error(f"Test RAG failed: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# Router Mounting Function
# ============================================================================

def mount_ml_routes(app):
    """
    Mount ML routes to FastAPI app
    
    Usage:
        from api.ml_routes import mount_ml_routes
        mount_ml_routes(app)
    """
    app.include_router(router, prefix="/api")
    logger.info("✅ ML RAG routes mounted at /api/v1/ml/*")
