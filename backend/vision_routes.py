import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.vision.vision_pipeline import process_and_analyze_image
from backend.vision.vision_rag import VisionRAG
from backend.vision.vision_twin_adapter import (
    adapt_disease_to_twin,
    adapt_nutrient_to_twin,
    adapt_health_to_twin,
    adapt_yield_to_twin
)
from backend.ollama_service import chat_with_agrigpt

logger = logging.getLogger("AgriVisionRoutes")
router = APIRouter(prefix="/api/vision", tags=["AgriVision"])

# Shared RAG instance
vision_rag = VisionRAG()


class VisionRequest(BaseModel):
    imageBase64: str
    mode: Optional[str] = "disease"


@router.post("/disease")
async def vision_disease(payload: VisionRequest):
    """Diagnoses crop diseases from leaf images, runs RAG advice, and syncs to twin."""
    res = await process_and_analyze_image(payload.imageBase64, mode="disease")
    if not res["success"]:
        raise HTTPException(status_code=500, detail=res.get("error", "Disease analysis failed"))

    # Run RAG
    res["results"] = vision_rag.augment_analysis(res["results"], query_key="disease")
    res["recommendations"] = res["results"].get("recommendations", [])

    # Sync Twin
    try:
        adapt_disease_to_twin(
            disease_name=res["results"].get("disease", "Unknown disease"),
            confidence=res["confidence"],
            severity=res["results"].get("severity", "medium")
        )
    except Exception as e:
        logger.error(f"Twin sync failed for disease: {e}")

    # AgriGPT plain-language explanation
    try:
        explanation = await chat_with_agrigpt(
            f"Explain this plant disease diagnosis in simple farmer-friendly terms: {res['results']}"
        )
        res["results"]["farmer_explanation"] = explanation
    except Exception:
        pass

    return res


@router.post("/weed")
async def vision_weed(payload: VisionRequest):
    """Identifies invasive weed species and triggers management recommendations."""
    res = await process_and_analyze_image(payload.imageBase64, mode="weed")
    if not res["success"]:
        raise HTTPException(status_code=500, detail=res.get("error", "Weed analysis failed"))

    # Run RAG with weed-specific knowledge
    res["results"] = vision_rag.augment_analysis(res["results"], query_key="weed")
    res["recommendations"] = res["results"].get("recommendations", [])

    return res


@router.post("/nutrient")
async def vision_nutrient(payload: VisionRequest):
    """Diagnoses soil NPK or micronutrient deficiencies from leaf visual markers."""
    res = await process_and_analyze_image(payload.imageBase64, mode="nutrient")
    if not res["success"]:
        raise HTTPException(status_code=500, detail=res.get("error", "Nutrient analysis failed"))

    # Run RAG with nutrient-specific knowledge
    res["results"] = vision_rag.augment_analysis(res["results"], query_key="nutrient")
    res["recommendations"] = res["results"].get("recommendations", [])

    deficiency = res["results"].get("disease", "Nitrogen Deficiency")
    severity = res["results"].get("severity", "medium")

    # Sync Twin
    try:
        adapt_nutrient_to_twin(deficiency=deficiency, severity=severity)
    except Exception as e:
        logger.error(f"Twin sync failed for nutrient: {e}")

    return res


@router.post("/pest")
async def vision_pest(payload: VisionRequest):
    """Detects insects, pests, and bugs eating leaf tissue."""
    res = await process_and_analyze_image(payload.imageBase64, mode="pest")
    if not res["success"]:
        raise HTTPException(status_code=500, detail=res.get("error", "Pest analysis failed"))

    # Run RAG with pest-specific knowledge
    res["results"] = vision_rag.augment_analysis(res["results"], query_key="pest")
    res["recommendations"] = res["results"].get("recommendations", [])

    return res


@router.post("/crop")
async def vision_crop(payload: VisionRequest):
    """Identifies crop type and estimates growth stage."""
    res = await process_and_analyze_image(payload.imageBase64, mode="crop")
    if not res["success"]:
        raise HTTPException(status_code=500, detail=res.get("error", "Crop identification failed"))

    # Run RAG with crop-specific knowledge
    res["results"] = vision_rag.augment_analysis(res["results"], query_key="crop")
    res["recommendations"] = res["results"].get("recommendations", [])

    return res


@router.post("/health")
async def vision_health(payload: VisionRequest):
    """Inspects chlorophyll vigor, NDVI proxy, and plant stress index."""
    res = await process_and_analyze_image(payload.imageBase64, mode="health")
    if not res["success"]:
        raise HTTPException(status_code=500, detail=res.get("error", "Health analysis failed"))

    # Run RAG with health-specific knowledge
    res["results"] = vision_rag.augment_analysis(res["results"], query_key="health")
    res["recommendations"] = res["results"].get("recommendations", [])

    # Sync Twin
    try:
        adapt_health_to_twin(
            health_score=res["confidence"],
            stress_level=res["results"].get("severity", "low")
        )
    except Exception as e:
        logger.error(f"Twin sync failed for health: {e}")

    return res


@router.post("/yield")
async def vision_yield(payload: VisionRequest):
    """Estimates crop yield potential from aerial or canopy pictures."""
    res = await process_and_analyze_image(payload.imageBase64, mode="yield")
    if not res["success"]:
        raise HTTPException(status_code=500, detail=res.get("error", "Yield estimation failed"))

    # Run RAG with yield-specific knowledge
    res["results"] = vision_rag.augment_analysis(res["results"], query_key="yield")
    res["recommendations"] = res["results"].get("recommendations", [])

    # Sync Twin
    try:
        adapt_yield_to_twin(estimated_yield=float(res["confidence"] * 5.0))
    except Exception as e:
        logger.error(f"Twin sync failed for yield: {e}")

    return res
