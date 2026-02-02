"""
AgriSense ML Prediction Routes
Endpoints for ML-based crop recommendations and predictions
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from core.engine import engine

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class CropRecommendationInput(BaseModel):
    temperature: float = Field(..., ge=0, le=50, description="Temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Humidity in %")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(default=100, ge=0, description="Rainfall in mm")
    nitrogen: Optional[float] = Field(default=40, description="Nitrogen in kg/ha")
    phosphorus: Optional[float] = Field(default=30, description="Phosphorus in kg/ha")
    potassium: Optional[float] = Field(default=35, description="Potassium in kg/ha")

class WaterOptimizationInput(BaseModel):
    crop: str
    growth_stage: str = Field(..., description="initial, mid, or late")
    temp_min: float
    temp_max: float
    soil_type: str = Field(default="loam", description="sandy, loam, or clay")

class FertilizerInput(BaseModel):
    crop: str
    soil_n: float = Field(..., description="Current nitrogen in kg/ha")
    soil_p: float = Field(..., description="Current phosphorus in kg/ha")
    soil_k: float = Field(..., description="Current potassium in kg/ha")
    soil_ph: float = Field(..., ge=0, le=14)

class YieldPredictionInput(BaseModel):
    crop: str
    area_hectares: float
    temperature: float
    humidity: float
    rainfall: float
    soil_type: str = Field(default="loam")

@router.post("/recommend")
async def recommend_crop(input_data: CropRecommendationInput):
    """
    Recommend best crops based on environmental conditions
    
    Uses rule-based engine with scoring algorithm
    """
    try:
        recommendations = engine.get_crop_recommendation(
            temperature=input_data.temperature,
            humidity=input_data.humidity,
            ph=input_data.ph,
            rainfall=input_data.rainfall
        )
        
        return {
            "success": True,
            "input": input_data.dict(),
            "recommendations": recommendations,
            "method": "rule-based"
        }
    except Exception as e:
        logger.error(f"Error in crop recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/water/optimize")
async def optimize_water(input_data: WaterOptimizationInput):
    """
    Calculate optimal water requirements using ET0 method
    
    Implements Hargreaves evapotranspiration formula
    """
    try:
        water_req = engine.get_water_requirement(
            crop=input_data.crop,
            growth_stage=input_data.growth_stage,
            temp_min=input_data.temp_min,
            temp_max=input_data.temp_max,
            soil_type=input_data.soil_type
        )
        
        return {
            "success": True,
            "input": input_data.dict(),
            "water_requirement": water_req,
            "method": "hargreaves_et0"
        }
    except Exception as e:
        logger.error(f"Error in water optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fertilizer/recommend")
async def recommend_fertilizer(input_data: FertilizerInput):
    """
    Calculate NPK fertilizer requirements
    
    Based on soil test results and crop requirements
    """
    try:
        fertilizer_rec = engine.get_fertilizer_recommendation(
            crop=input_data.crop,
            soil_n=input_data.soil_n,
            soil_p=input_data.soil_p,
            soil_k=input_data.soil_k,
            soil_ph=input_data.soil_ph
        )
        
        return {
            "success": True,
            "input": input_data.dict(),
            "fertilizer_recommendation": fertilizer_rec,
            "method": "deficit_calculation"
        }
    except Exception as e:
        logger.error(f"Error in fertilizer recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/yield/predict")
async def predict_yield(input_data: YieldPredictionInput):
    """
    Predict crop yield based on conditions
    
    Uses simplified yield estimation model
    """
    try:
        # Simplified yield prediction logic
        # In production, this would use trained ML models
        base_yields = {
            "Rice": 4.5,
            "Wheat": 3.8,
            "Maize": 5.2,
            "Cotton": 2.1,
            "Sugarcane": 70.0
        }
        
        base_yield = base_yields.get(input_data.crop, 3.0)
        
        # Adjust based on conditions (simplified)
        temp_factor = 1.0 if 20 <= input_data.temperature <= 30 else 0.85
        humidity_factor = 1.0 if 50 <= input_data.humidity <= 70 else 0.9
        rainfall_factor = min(1.0, input_data.rainfall / 800)
        
        predicted_yield = base_yield * temp_factor * humidity_factor * rainfall_factor
        total_yield = predicted_yield * input_data.area_hectares
        
        return {
            "success": True,
            "input": input_data.dict(),
            "prediction": {
                "yield_per_hectare": round(predicted_yield, 2),
                "total_yield_tonnes": round(total_yield, 2),
                "confidence": "medium",
                "factors": {
                    "temperature_factor": temp_factor,
                    "humidity_factor": humidity_factor,
                    "rainfall_factor": round(rainfall_factor, 2)
                }
            },
            "method": "simplified_model"
        }
    except Exception as e:
        logger.error(f"Error in yield prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/crops")
async def get_supported_crops():
    """
    Get list of supported crops
    
    Returns all crops in the database
    """
    try:
        crops = list(engine.crop_database.keys())
        
        return {
            "success": True,
            "count": len(crops),
            "crops": sorted(crops)
        }
    except Exception as e:
        logger.error(f"Error getting crops: {e}")
        # Fallback to default list
        return {
            "success": True,
            "count": 4,
            "crops": ["Rice", "Wheat", "Maize", "Cotton"]
        }

@router.get("/models/status")
async def get_models_status():
    """Get ML models status"""
    return {
        "success": True,
        "models": {
            "crop_recommendation": "rule-based",
            "water_optimization": "et0_hargreaves",
            "fertilizer": "deficit_calculation",
            "yield_prediction": "simplified"
        },
        "total_models": 4,
        "status": "operational"
    }
