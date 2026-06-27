import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..services.crop_service import crop_service

router = APIRouter()
logger = logging.getLogger(__name__)


class CropInput(BaseModel):
    pH: float = Field(..., ge=0, le=14, description="Soil pH level")
    N: float = Field(..., ge=0, description="Nitrogen content (kg/ha)")
    P: float = Field(..., ge=0, description="Phosphorus content (kg/ha)")
    K: float = Field(..., ge=0, description="Potassium content (kg/ha)")
    Fe: float = Field(..., ge=0, description="Iron content (ppm)")
    Mn: float = Field(..., ge=0, description="Manganese content (ppm)")
    Zn: float = Field(..., ge=0, description="Zinc content (ppm)")
    Cu: float = Field(..., ge=0, description="Copper content (ppm)")
    B: float = Field(..., ge=0, description="Boron content (ppm)")
    Water: float = Field(..., ge=0, description="Water availability (mm/season)")
    Moisture: float = Field(..., ge=0, le=100, description="Soil moisture percentage")
    Temperature: float = Field(..., description="Average temperature (Celsius)")
    Rainfall: float = Field(..., ge=0, description="Average rainfall (mm/season)")


class CropRecommendation(BaseModel):
    rank: int
    crop: str
    suitability: float


@router.post("/recommend", response_model=List[CropRecommendation])
async def recommend_crops(data: CropInput):
    """
    Get crop recommendations based on soil and environmental parameters.
    """
    try:
        # Convert Pydantic model to dict
        soil_data = data.model_dump()

        recommendations = crop_service.predict_crop(soil_data)
        return recommendations
    except ValueError as ve:
        raise HTTPException(status_code=503, detail=str(ve))
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during recommendation")
