"""
FastAPI Routes for Claude Enhanced Crop Recommendation System
Integrates with the main AgriSense AI service
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

from fastapi import APIRouter, HTTPException
import logging

from crop_recommender_api import (  # noqa: E402
    SoilParameters,
    CropRecommendationResponse,
    recommend_crops
)

router = APIRouter(
    prefix="/crop-recommendation",
    tags=["crop-recommendation"]
)
logger = logging.getLogger("CropRecommendationRouter")


@router.post("/predict", response_model=CropRecommendationResponse)
async def predict_crop_recommendation(params: SoilParameters):
    """
    Get crop recommendations based on soil parameters

    **Input Parameters (Soil Test Data):**
    - pH: Soil pH (4.5-8.5)
    - N: Nitrogen in kg/ha (10-300)
    - P: Phosphorus in kg/ha (10-150)
    - K: Potassium in kg/ha (10-200)
    - Fe: Iron in ppm (2.0-8.0)
    - Mn: Manganese in ppm (0.8-5.0)
    - Zn: Zinc in ppm (0.4-3.0)
    - Cu: Copper in ppm (0.15-1.5)
    - B: Boron in ppm (0.15-1.5)
    - Water: Water requirement in mm/season (200-2500)
    - Moisture: Soil moisture in % (35-90)
    - Temperature: Temperature in °C (10-40)
    - Rainfall: Rainfall in mm/season (200-2500)
    - top_n: Number of recommendations (1-10, default: 5)

    **Example Request:**
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

    **Returns:**
    - List of top N crop recommendations with suitability scores
    - Model information and metadata
    - Input parameters for reference
    """
    try:
        result = recommend_crops(params)
        return CropRecommendationResponse(**result)
    except ValueError as ve:
        logger.warning(f"Validation error in crop recommendation: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        error_msg = f"Error processing crop recommendation: {str(e)}"
        logger.error(f"Error in crop recommendation endpoint: {e}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/health")
async def health_check():
    """Check if crop recommendation service is operational"""
    return {
        "status": "operational",
        "service": "crop-recommendation",
        "model": "Random Forest with 100+ Indian crops"
    }


@router.get("/crops-list")
async def get_crops_list():
    """Get list of all supported crops"""
    from crop_requirements_dataset import crop_data

    crops = {
        "total": len(crop_data['Crop_Name']),
        "categories": {
            "Cereals": crop_data['Crop_Name'][0:15],
            "Pulses": crop_data['Crop_Name'][15:30],
            "Oilseeds": crop_data['Crop_Name'][30:45],
            "Cash Crops": crop_data['Crop_Name'][45:55],
            "Vegetables": crop_data['Crop_Name'][55:75],
            "Fruits": crop_data['Crop_Name'][75:90],
            "Spices": crop_data['Crop_Name'][90:100]
        }
    }
    return crops


@router.get("/crop-requirements/{crop_name}")
async def get_crop_requirements(crop_name: str):
    """Get soil requirements for a specific crop"""
    from crop_requirements_dataset import crop_data
    import pandas as pd

    df = pd.DataFrame(crop_data)
    crop_row = df[df['Crop_Name'].str.lower() == crop_name.lower()]

    if crop_row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Crop '{crop_name}' not found"
        )

    row = crop_row.iloc[0]
    requirements = {
        "crop_name": row['Crop_Name'],
        "ph_range": f"{row['pH_Min']:.1f} - {row['pH_Max']:.1f}",
        "nitrogen": f"{row['N_Min']:.0f} - {row['N_Max']:.0f} kg/ha",
        "phosphorus": f"{row['P_Min']:.0f} - {row['P_Max']:.0f} kg/ha",
        "potassium": f"{row['K_Min']:.0f} - {row['K_Max']:.0f} kg/ha",
        "water_requirement": (
            f"{row['Water_Requirement_Min']:.0f} - "
            f"{row['Water_Requirement_Max']:.0f} mm/season"
        ),
        "soil_moisture": (
            f"{row['Moisture_Min']:.0f} - {row['Moisture_Max']:.0f} %"
        ),
        "temperature_range": (
            f"{row['Temp_Min']:.0f} - {row['Temp_Max']:.0f} °C"
        ),
        "rainfall_range": (
            f"{row['Rainfall_Min']:.0f} - {row['Rainfall_Max']:.0f} mm/season"
        )
    }

    return requirements
