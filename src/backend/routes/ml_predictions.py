"""
Water Optimization and Yield Prediction API Routes for AgriSense
Provides ML-powered endpoints for irrigation optimization and crop yield prediction.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

# Configure logging
logger = logging.getLogger(__name__)

# Try to load ML components
try:
    import joblib
    import numpy as np
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available - ML models will not work")

# Initialize router
router = APIRouter(tags=["ML Predictions"])

# Model paths (relative to backend directory, not routes directory)
MODELS_DIR = Path(__file__).parent.parent / "models"

# Lazy-loaded model components
_water_model = None
_water_scaler = None
_water_crop_encoder = None
_water_soil_encoder = None

_yield_model = None
_yield_scaler = None
_yield_crop_encoder = None
_yield_soil_encoder = None


# ==================== PYDANTIC MODELS ====================

class WaterOptimizationRequest(BaseModel):
    """Request model for water optimization prediction"""
    soil_moisture: float = Field(50.0, ge=0, le=100, description="Current soil moisture percentage")
    temperature: float = Field(25.0, ge=-10, le=55, description="Ambient temperature in Celsius")
    humidity: float = Field(60.0, ge=0, le=100, description="Air humidity percentage")
    crop_type: str = Field("tomato", description="Type of crop")
    soil_type: str = Field("loam", description="Type of soil")
    evapotranspiration: float = Field(4.0, ge=0, le=15, description="Daily ET rate (mm/day)")
    rainfall_forecast: float = Field(0.0, ge=0, le=100, description="Expected rainfall in next 24h (mm)")
    plant_growth_stage: float = Field(0.5, ge=0, le=1, description="Growth stage (0-1 normalized)")
    area_m2: float = Field(100.0, gt=0, description="Area to irrigate in square meters")


class WaterOptimizationResponse(BaseModel):
    """Response model for water optimization prediction"""
    irrigation_volume_per_m2: float = Field(..., description="Recommended liters per square meter")
    total_irrigation_liters: float = Field(..., description="Total irrigation volume for the area")
    irrigation_urgency: float = Field(..., description="Urgency level (0-1)")
    recommended_frequency_days: int = Field(..., description="Recommended days between irrigation")
    confidence: float = Field(..., description="Model confidence (0-1)")
    model_version: str = Field(..., description="Model version used")
    recommendations: List[str] = Field(..., description="Additional irrigation tips")


class YieldPredictionRequest(BaseModel):
    """Request model for yield prediction"""
    crop_type: str = Field("corn", description="Type of crop")
    area_hectares: float = Field(1.0, gt=0, description="Cultivated area in hectares")
    nitrogen: float = Field(100.0, ge=0, le=500, description="Nitrogen content (kg/ha)")
    phosphorus: float = Field(40.0, ge=0, le=200, description="Phosphorus content (kg/ha)")
    potassium: float = Field(60.0, ge=0, le=300, description="Potassium content (kg/ha)")
    temperature: float = Field(25.0, ge=-10, le=55, description="Average temperature during growth")
    humidity: float = Field(65.0, ge=0, le=100, description="Average humidity percentage")
    rainfall: float = Field(300.0, ge=0, le=3000, description="Total rainfall during season (mm)")
    irrigation: float = Field(5000.0, ge=0, description="Total irrigation applied (liters/ha)")
    growing_days: int = Field(100, ge=30, le=400, description="Days from planting to harvest")
    soil_type: str = Field("loam", description="Type of soil")
    pest_pressure: float = Field(0.2, ge=0, le=1, description="Pest pressure level (0-1)")


class YieldPredictionResponse(BaseModel):
    """Response model for yield prediction"""
    predicted_yield_kg_ha: float = Field(..., description="Predicted yield in kg/hectare")
    total_production_kg: float = Field(..., description="Total production for the area in kg")
    yield_category: str = Field(..., description="Yield category (poor, below_average, average, good, excellent)")
    regional_average_yield: float = Field(..., description="Typical regional average yield in kg/ha")
    confidence: float = Field(..., description="Model confidence (0-1)")
    model_version: str = Field(..., description="Model version used")
    recommendations: List[str] = Field(..., description="Recommendations to improve yield")


# ==================== MODEL LOADING ====================

def _load_water_model():
    """Lazy load water optimization model components"""
    global _water_model, _water_scaler, _water_crop_encoder, _water_soil_encoder
    
    if not JOBLIB_AVAILABLE:
        logger.error("joblib not available")
        return False
        
    if _water_model is not None:
        return True
    
    try:
        model_path = MODELS_DIR / "water_model.joblib"
        scaler_path = MODELS_DIR / "water_scaler.joblib"
        crop_enc_path = MODELS_DIR / "water_crop_encoder.joblib"
        soil_enc_path = MODELS_DIR / "water_soil_encoder.joblib"
        
        if not model_path.exists():
            logger.error(f"Water model not found: {model_path}")
            return False
        
        _water_model = joblib.load(model_path)
        _water_scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        _water_crop_encoder = joblib.load(crop_enc_path) if crop_enc_path.exists() else None
        _water_soil_encoder = joblib.load(soil_enc_path) if soil_enc_path.exists() else None
        
        logger.info("✅ Water optimization model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load water model: {e}")
        return False


def _load_yield_model():
    """Lazy load yield prediction model components"""
    global _yield_model, _yield_scaler, _yield_crop_encoder, _yield_soil_encoder
    
    if not JOBLIB_AVAILABLE:
        logger.error("joblib not available")
        return False
        
    if _yield_model is not None:
        return True
    
    try:
        model_path = MODELS_DIR / "yield_prediction_model.joblib"
        scaler_path = MODELS_DIR / "yield_scaler.joblib"
        crop_enc_path = MODELS_DIR / "yield_crop_encoder.joblib"
        soil_enc_path = MODELS_DIR / "yield_soil_encoder.joblib"
        
        if not model_path.exists():
            logger.error(f"Yield model not found: {model_path}")
            return False
        
        _yield_model = joblib.load(model_path)
        _yield_scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        _yield_crop_encoder = joblib.load(crop_enc_path) if crop_enc_path.exists() else None
        _yield_soil_encoder = joblib.load(soil_enc_path) if soil_enc_path.exists() else None
        
        logger.info("✅ Yield prediction model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load yield model: {e}")
        return False


# ==================== CROP/SOIL CONFIGURATIONS ====================

WATER_CROP_CONFIGS = {
    'almond': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'apple': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'arecanut': {'base_water': 7.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'arhar': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'bajra': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'banana': {'base_water': 8.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'barley': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'barnyard_millet': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'beetroot': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'bitter_gourd': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'black_pepper': {'base_water': 6.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'bottle_gourd': {'base_water': 5.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'brinjal': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'buckwheat': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'cabbage': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'cardamom': {'base_water': 6.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'carrot': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'cashew': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'castor': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'cauliflower': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'chickpea': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'chilli': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'cluster_bean': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'coconut': {'base_water': 6.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'coffee': {'base_water': 7.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'coriander': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'cotton': {'base_water': 7.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'cucumber': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'cumin': {'base_water': 2.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'custard_apple': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'dragon_fruit': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'fenugreek': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'field_pea': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'foxtail_millet': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'french_bean': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'garlic': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'ginger': {'base_water': 5.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'grapes': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'green_pea': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'groundnut': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'guava': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'horse_gram': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'jackfruit': {'base_water': 6.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'jowar': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'jute': {'base_water': 8.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'kidney_bean': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'kodo_millet': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'lentil': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'lettuce': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'linseed': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'litchi': {'base_water': 6.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'little_millet': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'maize': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'mango': {'base_water': 6.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'masoor': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'moong': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'moth_bean': {'base_water': 2.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'muskmelon': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'mustard': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'niger': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'oats': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'okra': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'onion': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'orange': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'papaya': {'base_water': 6.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'pearl_millet': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'pigeon_pea': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'pineapple': {'base_water': 5.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'pomegranate': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'potato': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'proso_millet': {'base_water': 2.8, 'critical_moisture': 30, 'optimal_moisture': 60},
    'pumpkin': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'radish': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'ragi': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'rice': {'base_water': 9.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'ridge_gourd': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'rubber': {'base_water': 8.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'safflower': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'sapota': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'sesame': {'base_water': 3.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'sorghum': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'soybean': {'base_water': 5.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'spinach': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'strawberry': {'base_water': 4.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'sugarcane': {'base_water': 15.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'sunflower': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'sweet_potato': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'tea': {'base_water': 8.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'tobacco': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'tomato': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'turmeric': {'base_water': 5.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'turnip': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'urad': {'base_water': 3.5, 'critical_moisture': 30, 'optimal_moisture': 60},
    'walnut': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'watermelon': {'base_water': 5.0, 'critical_moisture': 30, 'optimal_moisture': 60},
    'wheat': {'base_water': 4.5, 'critical_moisture': 30, 'optimal_moisture': 60},
}

YIELD_CROP_CONFIGS = {
    'almond': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 22.5, 'growth_days': 365},
    'apple': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 15.0, 'growth_days': 365},
    'arecanut': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 365},
    'arhar': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 180},
    'bajra': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 26.0, 'growth_days': 70},
    'banana': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 300},
    'barley': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 18.5, 'growth_days': 110},
    'barnyard_millet': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 60},
    'beetroot': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 90},
    'bitter_gourd': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 29.5, 'growth_days': 55},
    'black_pepper': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 365},
    'bottle_gourd': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.0, 'growth_days': 60},
    'brinjal': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 120},
    'buckwheat': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 90},
    'cabbage': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 17.5, 'growth_days': 90},
    'cardamom': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 365},
    'carrot': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 90},
    'cashew': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 365},
    'castor': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 150},
    'cauliflower': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 75},
    'chickpea': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 120},
    'chilli': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 150},
    'cluster_bean': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 75},
    'coconut': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 365},
    'coffee': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 23.0, 'growth_days': 365},
    'coriander': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 90},
    'cotton': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.5, 'growth_days': 150},
    'cucumber': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 24.0, 'growth_days': 50},
    'cumin': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 120},
    'custard_apple': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 365},
    'dragon_fruit': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 365},
    'fenugreek': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 90},
    'field_pea': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 17.5, 'growth_days': 100},
    'foxtail_millet': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 70},
    'french_bean': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 22.5, 'growth_days': 60},
    'garlic': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 18.5, 'growth_days': 150},
    'ginger': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 240},
    'grapes': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 365},
    'green_pea': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 18.5, 'growth_days': 90},
    'groundnut': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 110},
    'guava': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 365},
    'horse_gram': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 120},
    'jackfruit': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 365},
    'jowar': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 28.5, 'growth_days': 100},
    'jute': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 120},
    'kidney_bean': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 90},
    'kodo_millet': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 90},
    'lentil': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 23.0, 'growth_days': 110},
    'lettuce': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 60},
    'linseed': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 120},
    'litchi': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 365},
    'little_millet': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 90},
    'maize': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 22.5, 'growth_days': 90},
    'mango': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.0, 'growth_days': 365},
    'masoor': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 120},
    'moong': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 60},
    'moth_bean': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 33.0, 'growth_days': 75},
    'muskmelon': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 90},
    'mustard': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 17.5, 'growth_days': 110},
    'niger': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 110},
    'oats': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 17.5, 'growth_days': 120},
    'okra': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 28.5, 'growth_days': 60},
    'onion': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 120},
    'orange': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 365},
    'papaya': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 26.0, 'growth_days': 365},
    'pearl_millet': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 75},
    'pigeon_pea': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 150},
    'pineapple': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 18},
    'pomegranate': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 365},
    'potato': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 90},
    'proso_millet': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 23.0, 'growth_days': 70},
    'pumpkin': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 26.0, 'growth_days': 90},
    'radish': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 23.0, 'growth_days': 45},
    'ragi': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 120},
    'rice': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 120},
    'ridge_gourd': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 65},
    'rubber': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 365},
    'safflower': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 120},
    'sapota': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 365},
    'sesame': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 85},
    'sorghum': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 29.5, 'growth_days': 100},
    'soybean': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 90},
    'spinach': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 45},
    'strawberry': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 150},
    'sugarcane': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 300},
    'sunflower': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 22.5, 'growth_days': 90},
    'sweet_potato': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 25.0, 'growth_days': 110},
    'tea': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 22.5, 'growth_days': 365},
    'tobacco': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 21.0, 'growth_days': 120},
    'tomato': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 22.5, 'growth_days': 90},
    'turmeric': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 27.5, 'growth_days': 240},
    'turnip': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 20.0, 'growth_days': 60},
    'urad': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 30.0, 'growth_days': 70},
    'walnut': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 17.5, 'growth_days': 365},
    'watermelon': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 29.5, 'growth_days': 90},
    'wheat': {'min_yield': 2000, 'max_yield': 6000, 'optimal_temp': 17.5, 'growth_days': 120},
}

SOIL_CONFIGS = {
    'sandy': {'retention': 0.6, 'drainage': 1.4},
    'loam': {'retention': 1.0, 'drainage': 1.0},
    'clay': {'retention': 1.3, 'drainage': 0.7},
    'clay_loam': {'retention': 1.15, 'drainage': 0.85},
    'sandy_loam': {'retention': 0.8, 'drainage': 1.2},
    'silty': {'retention': 1.1, 'drainage': 0.9},
    'black_cotton': {'fertility': 1.1, 'nutrient_retention': 1.05},
}


# ==================== API ENDPOINTS ====================

@router.post("/water-optimization", response_model=WaterOptimizationResponse)
async def predict_water_optimization(request: WaterOptimizationRequest) -> WaterOptimizationResponse:
    """
    Predict optimal irrigation volume and scheduling based on environmental conditions.
    
    This endpoint uses a Random Forest ML model trained on agricultural data to
    provide irrigation recommendations considering:
    - Current soil moisture levels
    - Weather conditions (temperature, humidity)
    - Crop-specific water requirements
    - Soil water retention characteristics
    - Upcoming rainfall forecasts
    
    Returns irrigation volume recommendations and scheduling advice.
    """
    
    # Load model if needed
    if not _load_water_model():
        # Fallback to rule-based estimation
        return _fallback_water_prediction(request)
    
    try:
        # Normalize crop and soil types
        crop = request.crop_type.lower().strip()
        soil = request.soil_type.lower().strip().replace(' ', '_')
        
        # Default to known types if unknown
        if crop not in WATER_CROP_CONFIGS:
            crop = 'tomato'
        if soil not in SOIL_CONFIGS:
            soil = 'loam'
        
        # Encode categorical features
        crop_encoded = 0
        soil_encoded = 0
        
        if _water_crop_encoder is not None:
            try:
                crop_encoded = _water_crop_encoder.transform([crop])[0]
            except ValueError:
                crop_encoded = 0
                
        if _water_soil_encoder is not None:
            try:
                soil_encoded = _water_soil_encoder.transform([soil])[0]
            except ValueError:
                soil_encoded = 0
        
        # Prepare feature vector
        features = np.array([[
            request.soil_moisture,
            request.temperature,
            request.humidity,
            crop_encoded,
            soil_encoded,
            request.evapotranspiration,
            request.rainfall_forecast,
            request.plant_growth_stage
        ]])
        
        # Scale features
        if _water_scaler is not None:
            features_scaled = _water_scaler.transform(features)
        else:
            features_scaled = features
        
        # Make prediction
        prediction = _water_model.predict(features_scaled)[0]
        prediction = max(0, float(prediction))
        
        # Calculate derived values
        total_liters = prediction * request.area_m2
        
        # Calculate urgency based on moisture deficit
        crop_config = WATER_CROP_CONFIGS.get(crop, WATER_CROP_CONFIGS['tomato'])
        moisture_deficit = max(0, crop_config['optimal_moisture'] - request.soil_moisture)
        urgency = moisture_deficit / 60  # Normalize
        urgency = min(1.0, max(0.0, urgency))
        
        # Calculate recommended frequency
        frequency = max(1, round(3 / (urgency + 0.1)))
        
        # Generate recommendations
        recommendations = _generate_water_recommendations(request, prediction, urgency, crop_config)
        
        return WaterOptimizationResponse(
            irrigation_volume_per_m2=round(prediction, 2),
            total_irrigation_liters=round(total_liters, 1),
            irrigation_urgency=round(urgency, 2),
            recommended_frequency_days=frequency,
            confidence=0.82,
            model_version="1.0.0-rf",
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Water optimization prediction error: {e}")
        return _fallback_water_prediction(request)


@router.post("/yield-prediction", response_model=YieldPredictionResponse)
async def predict_yield(request: YieldPredictionRequest) -> YieldPredictionResponse:
    """
    Predict crop yield based on environmental conditions and farming practices.
    
    This endpoint uses a Gradient Boosting ML model trained on agricultural data
    to predict yields considering:
    - Crop type and growth characteristics
    - Soil nutrient levels (N, P, K)
    - Weather conditions
    - Irrigation practices
    - Pest pressure
    
    Returns yield predictions in kg/hectare with comparison to typical yields.
    """
    
    # Load model if needed
    if not _load_yield_model():
        # Fallback to rule-based estimation
        return _fallback_yield_prediction(request)
    
    try:
        # Normalize crop and soil types
        crop = request.crop_type.lower().strip()
        soil = request.soil_type.lower().strip().replace(' ', '_')
        
        # Default to known types if unknown
        if crop not in YIELD_CROP_CONFIGS:
            crop = 'corn'
        if soil not in SOIL_CONFIGS:
            soil = 'loam'
        
        # Encode categorical features
        crop_encoded = 0
        soil_encoded = 0
        
        if _yield_crop_encoder is not None:
            try:
                crop_encoded = _yield_crop_encoder.transform([crop])[0]
            except ValueError:
                crop_encoded = 0
                
        if _yield_soil_encoder is not None:
            try:
                soil_encoded = _yield_soil_encoder.transform([soil])[0]
            except ValueError:
                soil_encoded = 0
        
        # Prepare feature vector
        features = np.array([[
            crop_encoded,
            request.area_hectares,
            request.nitrogen,
            request.phosphorus,
            request.potassium,
            request.temperature,
            request.humidity,
            request.rainfall,
            request.irrigation,
            request.growing_days,
            soil_encoded,
            request.pest_pressure
        ]])
        
        # Scale features
        if _yield_scaler is not None:
            features_scaled = _yield_scaler.transform(features)
        else:
            features_scaled = features
        
        # Make prediction
        yield_per_ha = _yield_model.predict(features_scaled)[0]
        yield_per_ha = max(0, float(yield_per_ha))
        
        # Calculate derived values
        total_production = yield_per_ha * request.area_hectares
        
        # Get crop-specific comparison data
        crop_config = YIELD_CROP_CONFIGS.get(crop, YIELD_CROP_CONFIGS['corn'])
        avg_yield = (crop_config['min_yield'] + crop_config['max_yield']) / 2
        
        # Determine yield category
        if yield_per_ha >= crop_config['max_yield'] * 0.9:
            yield_category = "excellent"
        elif yield_per_ha >= avg_yield * 1.1:
            yield_category = "good"
        elif yield_per_ha >= avg_yield * 0.9:
            yield_category = "average"
        elif yield_per_ha >= crop_config['min_yield'] * 1.2:
            yield_category = "below_average"
        else:
            yield_category = "poor"
        
        # Generate recommendations
        recommendations = _generate_yield_recommendations(request, yield_per_ha, crop_config)
        
        return YieldPredictionResponse(
            predicted_yield_kg_ha=round(yield_per_ha, 0),
            total_production_kg=round(total_production, 0),
            yield_category=yield_category,
            regional_average_yield=round(avg_yield, 0),
            confidence=0.85,
            model_version="1.0.0-gb",
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Yield prediction error: {e}")
        return _fallback_yield_prediction(request)


@router.get("/water-optimization/info")
async def water_optimization_info() -> Dict[str, Any]:
    """Get information about the water optimization model and supported crops/soils."""
    model_loaded = _water_model is not None or _load_water_model()
    
    return {
        "model_available": model_loaded,
        "model_version": "1.0.0-rf" if model_loaded else None,
        "supported_crops": list(WATER_CROP_CONFIGS.keys()),
        "supported_soils": list(SOIL_CONFIGS.keys()),
        "feature_descriptions": {
            "soil_moisture": "Current soil moisture percentage (0-100)",
            "temperature": "Ambient temperature in Celsius",
            "humidity": "Air humidity percentage (0-100)",
            "crop_type": "Type of crop being irrigated",
            "soil_type": "Type of soil in the field",
            "evapotranspiration": "Daily ET rate (mm/day)",
            "rainfall_forecast": "Expected rainfall in next 24 hours (mm)",
            "plant_growth_stage": "Growth stage (0-1, where 0=seedling, 1=mature)",
            "area_m2": "Area to irrigate in square meters"
        }
    }


@router.get("/yield-prediction/info")
async def yield_prediction_info() -> Dict[str, Any]:
    """Get information about the yield prediction model and supported crops/soils."""
    model_loaded = _yield_model is not None or _load_yield_model()
    
    return {
        "model_available": model_loaded,
        "model_version": "1.0.0-gb" if model_loaded else None,
        "supported_crops": list(YIELD_CROP_CONFIGS.keys()),
        "supported_soils": list(SOIL_CONFIGS.keys()),
        "crop_typical_yields": {
            crop: {
                "min_yield_kg_ha": config["min_yield"],
                "max_yield_kg_ha": config["max_yield"],
                "optimal_temp": config["optimal_temp"],
                "typical_growth_days": config["growth_days"]
            }
            for crop, config in YIELD_CROP_CONFIGS.items()
        }
    }


# ==================== FALLBACK FUNCTIONS ====================

def _fallback_water_prediction(request: WaterOptimizationRequest) -> WaterOptimizationResponse:
    """Rule-based fallback for water optimization when ML model unavailable"""
    
    crop = request.crop_type.lower().strip()
    crop_config = WATER_CROP_CONFIGS.get(crop, WATER_CROP_CONFIGS['tomato'])
    
    # Simple rule-based calculation
    base_need = crop_config['base_water']
    moisture_deficit = max(0, crop_config['optimal_moisture'] - request.soil_moisture)
    
    # Adjust for conditions
    temp_factor = 1 + (request.temperature - 25) * 0.02
    humidity_factor = 1 - (request.humidity - 50) * 0.005
    rain_reduction = min(0.8, request.rainfall_forecast / 10)
    
    irrigation_volume = base_need * (1 + moisture_deficit / 50) * temp_factor * humidity_factor * (1 - rain_reduction)
    irrigation_volume = max(0, irrigation_volume)
    
    urgency = moisture_deficit / 60
    frequency = max(1, round(3 / (urgency + 0.1)))
    
    return WaterOptimizationResponse(
        irrigation_volume_per_m2=round(irrigation_volume, 2),
        total_irrigation_liters=round(irrigation_volume * request.area_m2, 1),
        irrigation_urgency=round(min(1.0, urgency), 2),
        recommended_frequency_days=frequency,
        confidence=0.60,
        model_version="fallback-rules",
        recommendations=[
            "⚠️ Using rule-based estimation (ML model unavailable)",
            f"Target soil moisture: {crop_config['optimal_moisture']}%",
            f"Current deficit: {moisture_deficit}%"
        ]
    )


def _fallback_yield_prediction(request: YieldPredictionRequest) -> YieldPredictionResponse:
    """Rule-based fallback for yield prediction when ML model unavailable"""
    
    crop = request.crop_type.lower().strip()
    crop_config = YIELD_CROP_CONFIGS.get(crop, YIELD_CROP_CONFIGS['corn'])
    
    # Simple rule-based calculation
    avg_yield = (crop_config['min_yield'] + crop_config['max_yield']) / 2
    
    # Temperature factor
    temp_diff = abs(request.temperature - crop_config['optimal_temp'])
    temp_factor = max(0.5, 1 - temp_diff * 0.03)
    
    # Nutrient factor
    nutrient_score = (request.nitrogen / 150 + request.phosphorus / 60 + request.potassium / 100) / 3
    nutrient_factor = min(1.2, 0.6 + 0.5 * nutrient_score)
    
    # Calculate yield
    yield_per_ha = avg_yield * temp_factor * nutrient_factor * (1 - request.pest_pressure * 0.3)
    total_production = yield_per_ha * request.area_hectares
    
    # Determine yield category
    if yield_per_ha >= crop_config['max_yield'] * 0.9:
        yield_category = "excellent"
    elif yield_per_ha >= avg_yield * 1.1:
        yield_category = "good"
    elif yield_per_ha >= avg_yield * 0.9:
        yield_category = "average"
    elif yield_per_ha >= crop_config['min_yield'] * 1.2:
        yield_category = "below_average"
    else:
        yield_category = "poor"
    
    # Generate recommendations
    recommendations = _generate_yield_recommendations(request, yield_per_ha, crop_config)
    recommendations.insert(0, "⚠️ Using rule-based estimation (ML model unavailable)")
    
    return YieldPredictionResponse(
        predicted_yield_kg_ha=round(yield_per_ha, 0),
        total_production_kg=round(total_production, 0),
        yield_category=yield_category,
        regional_average_yield=round(avg_yield, 0),
        confidence=0.55,
        model_version="fallback-rules",
        recommendations=recommendations[:5]
    )


def _generate_water_recommendations(
    request: WaterOptimizationRequest,
    prediction: float,
    urgency: float,
    crop_config: dict
) -> List[str]:
    """Generate contextual irrigation recommendations"""
    recommendations = []
    
    # Urgency-based recommendations
    if urgency > 0.7:
        recommendations.append("⚠️ High irrigation urgency - water immediately")
    elif urgency > 0.4:
        recommendations.append("💧 Moderate irrigation needed within 24-48 hours")
    else:
        recommendations.append("✅ Soil moisture is adequate")
    
    # Temperature-based
    if request.temperature > 35:
        recommendations.append("🌡️ High temperature - consider evening irrigation to reduce evaporation")
    elif request.temperature < 15:
        recommendations.append("❄️ Cool conditions - reduce irrigation frequency")
    
    # Rainfall-based
    if request.rainfall_forecast > 5:
        recommendations.append(f"🌧️ {request.rainfall_forecast}mm rain expected - irrigation reduced accordingly")
    
    # Growth stage-based
    if request.plant_growth_stage < 0.2:
        recommendations.append("🌱 Seedling stage - maintain consistent moisture")
    elif request.plant_growth_stage > 0.8:
        recommendations.append("🌾 Maturation stage - reduce water for quality")
    
    return recommendations[:4]  # Limit to 4 recommendations


def _generate_yield_recommendations(
    request: YieldPredictionRequest,
    yield_per_ha: float,
    crop_config: dict
) -> List[str]:
    """Generate contextual yield improvement recommendations"""
    recommendations = []
    avg_yield = (crop_config['min_yield'] + crop_config['max_yield']) / 2
    
    # Overall yield assessment
    if yield_per_ha >= avg_yield * 1.1:
        recommendations.append("✅ Yield projection is above average - maintain current practices")
    elif yield_per_ha < avg_yield * 0.9:
        recommendations.append("⚠️ Yield projection is below average - consider optimization")
    
    # NPK recommendations
    total_npk = request.nitrogen + request.phosphorus + request.potassium
    if request.nitrogen < 60:
        recommendations.append("Consider increasing nitrogen application for better vegetative growth")
    elif request.nitrogen > 150:
        recommendations.append("⚠️ High nitrogen may cause lodging - consider reducing")
    
    if request.phosphorus < 30:
        recommendations.append("Low phosphorus levels - may limit root development and flowering")
    
    if request.potassium < 40:
        recommendations.append("Increase potassium for better disease resistance and fruit quality")
    
    # Temperature recommendations
    optimal_temp = crop_config.get('optimal_temp', 25)
    if abs(request.temperature - optimal_temp) > 8:
        recommendations.append(f"🌡️ Temperature ({request.temperature}°C) differs from optimal ({optimal_temp}°C)")
    
    # Water recommendations
    total_water = request.rainfall + request.irrigation
    if total_water < 500:
        recommendations.append("💧 Consider increasing irrigation - total water appears low")
    elif total_water > 1500:
        recommendations.append("⚠️ High water input may cause waterlogging issues")
    
    # Pest pressure
    if request.pest_pressure > 0.5:
        recommendations.append("🐛 High pest pressure detected - implement pest management")
    elif request.pest_pressure > 0.3:
        recommendations.append("Monitor for pest activity - moderate pressure detected")
    
    # Growing days
    expected_days = crop_config.get('growth_days', 100)
    if request.growing_days < expected_days * 0.8:
        recommendations.append(f"Growing period ({request.growing_days} days) may be short for {request.crop_type}")
    
    return recommendations[:5]  # Limit to 5 recommendations


def _get_yield_rating(yield_per_ha: float, crop_config: dict) -> str:
    """Get a qualitative yield rating"""
    avg = (crop_config['min_yield'] + crop_config['max_yield']) / 2
    
    if yield_per_ha >= crop_config['max_yield'] * 0.9:
        return "Excellent - Near maximum potential"
    elif yield_per_ha >= avg * 1.1:
        return "Very Good - Above average"
    elif yield_per_ha >= avg * 0.9:
        return "Good - Average yield expected"
    elif yield_per_ha >= crop_config['min_yield'] * 1.2:
        return "Fair - Below average"
    else:
        return "Poor - Consider adjusting conditions"
