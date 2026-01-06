from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
from ..ai.smart_recommendations import (
    WaterUsageEstimator,
    YieldPredictor,
)

router = APIRouter()


@router.post("/v1/recommendations/irrigation")
async def recommend_irrigation(payload: Dict[str, Any]):
    try:
        field_size = float(payload.get("field_size", 1.0))
        crop_type = payload.get("crop_type", "tomato")
        temperature = float(payload.get("temperature", 25.0))
        humidity = float(payload.get("humidity", 50.0))
        soil_moisture = float(payload.get("current_moisture", payload.get("soil_moisture", 50.0)))

        liters = WaterUsageEstimator.estimate_water_usage(
            crop_type=crop_type,
            temperature=temperature,
            humidity=humidity,
            soil_moisture=soil_moisture,
            field_size=field_size,
        )

        return {
            "recommendations": [
                {
                    "type": "irrigation",
                    "daily_water_liters": round(liters, 1),
                    "note": "Rule-based estimate (fallback). Replace with ML engine for production.",
                }
            ],
            "objectives": {"water_liters": round(liters, 1)},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/recommendations/fertilizer")
async def recommend_fertilizer(payload: Dict[str, Any]):
    try:
        field_size = float(payload.get("field_size", 1.0))
        # current nutrient levels (kg/ha)
        n = float(payload.get("current_n", 0.0))
        p = float(payload.get("current_p", 0.0))
        k = float(payload.get("current_k", 0.0))

        # Simple heuristic: recommend amounts to reach baseline targets
        targets = {"n": 150.0, "p": 80.0, "k": 150.0}
        rec_n = max(0.0, targets["n"] - n)
        rec_p = max(0.0, targets["p"] - p)
        rec_k = max(0.0, targets["k"] - k)

        return {
            "recommendations": [
                {
                    "type": "fertilizer",
                    "nitrogen_kg_per_ha": round(rec_n, 1),
                    "phosphorus_kg_per_ha": round(rec_p, 1),
                    "potassium_kg_per_ha": round(rec_k, 1),
                    "note": "Rule-based fallback recommendations.",
                }
            ],
            "objectives": {"n": rec_n, "p": rec_p, "k": rec_k},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/predict/explain")
async def predict_explain(payload: Dict[str, Any]):
    try:
        temp = float(payload.get("temperature", 25.0))
        humidity = float(payload.get("humidity", 50.0))
        soil_moisture = float(payload.get("soil_moisture", 50.0))
        ph = float(payload.get("ph_level", 6.5))
        n = float(payload.get("nitrogen", 100.0))
        p = float(payload.get("phosphorus", 50.0))
        k = float(payload.get("potassium", 100.0))

        pred = YieldPredictor.predict_yield(
            crop_type=payload.get("crop_type", "tomato"),
            soil_moisture=soil_moisture,
            temperature=temp,
            nitrogen=n,
            phosphorus=p,
            potassium=k,
            field_size=float(payload.get("field_size", 1.0)),
        )

        explanation = {
            "prediction": round(pred, 1),
            "explanation": {
                "drivers": [
                    {"feature": "soil_moisture", "value": soil_moisture},
                    {"feature": "temperature", "value": temp},
                    {"feature": "nitrogen", "value": n},
                ],
                "note": "Rule-based explanation (fallback).",
            },
        }
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vlm/status")
async def vlm_status():
    # Provide a lightweight status so E2E tests can succeed even if VLM is disabled
    return {"status": "disabled", "details": "VLM engine not initialized in this runtime"}
