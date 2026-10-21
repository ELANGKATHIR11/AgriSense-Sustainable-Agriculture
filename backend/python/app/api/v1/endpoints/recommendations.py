import os
import json
from typing import Optional
from fastapi import APIRouter, Query
from app.ml.crop_recommender import crop_recommender
from app.ml.fertilizer_engine import fertilizer_engine
from app.schemas.recommendations import (
    CropRecommendationRequestSchema, CropRecommendationResponseSchema,
    FertilizerRequestSchema, FertilizerResponseSchema
)
from app.core.config import settings

router = APIRouter()

# Load 200 crops from json file
CROPS_JSON_PATH = os.path.join(settings.BASE_DIR, "backend", "python", "app", "data", "crops_200.json")

def load_crops_catalog():
    if os.path.exists(CROPS_JSON_PATH):
        with open(CROPS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

ALL_CROPS = load_crops_catalog()

@router.get("/crops")
def get_all_crops(
    search: Optional[str] = Query(None, description="Search term for crop name or scientific name"),
    category: Optional[str] = Query(None, description="Category filter")
):
    crops = ALL_CROPS
    if category and category != "all":
        crops = [c for c in crops if c.get("category", "").lower() == category.lower()]
    if search:
        s = search.lower()
        crops = [
            c for c in crops
            if s in c.get("name", "").lower()
            or s in c.get("scientificName", "").lower()
            or s in c.get("majorStates", "").lower()
            or s in c.get("season", "").lower()
        ]
    
    cats = sorted(list(set(c["category"] for c in ALL_CROPS)))
    return {
        "total": len(crops),
        "totalCatalog": len(ALL_CROPS),
        "categories": cats,
        "crops": crops
    }

@router.post("/crop", response_model=CropRecommendationResponseSchema)
def recommend_crop(payload: CropRecommendationRequestSchema):
    recs = crop_recommender.recommend(
        n=payload.nitrogen,
        p=payload.phosphorus,
        k=payload.potassium,
        temp=payload.temperature,
        hum=payload.humidity,
        ph=payload.ph,
        rain=payload.rainfall
    )
    return {
        "recommendations": recs,
        "modelSource": "AgriSense-NPK-Suitability-Engine:v2.2.0 (ICAR Agronomic Calibration)"
    }

@router.post("/fertilizer", response_model=FertilizerResponseSchema)
def recommend_fertilizer(payload: FertilizerRequestSchema):
    result = fertilizer_engine.calculate(
        crop_type=payload.cropType,
        n=payload.nitrogen,
        p=payload.phosphorus,
        k=payload.potassium,
        soil_ph=payload.soilPh,
        moisture=payload.soilMoisture
    )
    return result
