"""
Claude-Enhanced Crop Recommendation API
FastAPI endpoint for intelligent crop recommendations based on soil parameters
"""

import sys
import os

# Add paths for imports before other imports
sys.path.append(os.path.dirname(__file__))

from typing import Optional, List
import logging
import pandas as pd
from pydantic import BaseModel, Field

from crop_recommendation_ml_model import CropRecommendationSystem  # noqa: E402
from crop_requirements_dataset import crop_data  # noqa: E402

# Setup logging
logger = logging.getLogger("CropRecommender")


class SoilParameters(BaseModel):
    """Input soil parameters for crop recommendation"""
    pH: float = Field(..., ge=4.5, le=8.5, description="Soil pH (4.5-8.5)")
    N: float = Field(..., ge=10, le=300, description="Nitrogen in kg/ha")
    P: float = Field(..., ge=10, le=150, description="Phosphorus in kg/ha")
    K: float = Field(..., ge=10, le=200, description="Potassium in kg/ha")
    Fe: float = Field(..., ge=2.0, le=8.0, description="Iron in ppm")
    Mn: float = Field(..., ge=0.8, le=5.0, description="Manganese in ppm")
    Zn: float = Field(..., ge=0.4, le=3.0, description="Zinc in ppm")
    Cu: float = Field(..., ge=0.15, le=1.5, description="Copper in ppm")
    B: float = Field(..., ge=0.15, le=1.5, description="Boron in ppm")
    Water: float = Field(
        ...,
        ge=200,
        le=2500,
        description="Water requirement in mm/season"
    )
    Moisture: float = Field(
        ..., ge=35, le=90, description="Soil moisture in %"
    )
    Temperature: float = Field(
        ...,
        ge=10,
        le=40,
        description="Temperature in Celsius"
    )
    Rainfall: float = Field(
        ...,
        ge=200,
        le=2500,
        description="Rainfall in mm/season"
    )
    top_n: Optional[int] = Field(
        5,
        ge=1,
        le=10,
        description="Number of recommendations"
    )


class CropRecommendation(BaseModel):
    """Single crop recommendation"""
    rank: int
    crop_name: str
    suitability_score: float
    confidence: str


class CropRecommendationResponse(BaseModel):
    """API Response for crop recommendations"""
    success: bool
    recommendations: List[CropRecommendation]
    model_info: dict
    input_parameters: dict


# Global recommendation system instance
_crop_recommender = None


def get_crop_recommender():
    """Initialize or return the crop recommender instance"""
    global _crop_recommender

    if _crop_recommender is None:
        logger.info("Initializing Crop Recommendation System...")
        _crop_recommender = CropRecommendationSystem()

        # Check if model exists, otherwise train
        model_path = os.path.join(
            os.path.dirname(__file__),
            'crop_recommendation_model.pkl'
        )

        if os.path.exists(model_path):
            logger.info("Loading pre-trained crop recommendation model...")
            _crop_recommender.load_model(model_path)
        else:
            logger.info("Training new crop recommendation model...")
            crop_df = pd.DataFrame(crop_data)
            accuracy = _crop_recommender.train(crop_df)
            _crop_recommender.save_model(model_path)
            logger.info(f"Model trained with accuracy: {accuracy:.4f}")

    return _crop_recommender


def recommend_crops(params: SoilParameters) -> dict:
    """
    Get crop recommendations based on soil parameters

    Args:
        params: SoilParameters object with soil test data

    Returns:
        dict: Recommendations response
    """
    try:
        recommender = get_crop_recommender()

        soil_data = {
            'pH': params.pH,
            'N': params.N,
            'P': params.P,
            'K': params.K,
            'Fe': params.Fe,
            'Mn': params.Mn,
            'Zn': params.Zn,
            'Cu': params.Cu,
            'B': params.B,
            'Water': params.Water,
            'Moisture': params.Moisture,
            'Temperature': params.Temperature,
            'Rainfall': params.Rainfall,
        }

        # Get recommendations
        recommendations = recommender.predict_crop(
            soil_data,
            top_n=params.top_n or 5
        )

        response = {
            'success': True,
            'recommendations': recommendations,
            'model_info': {
                'model_name': recommender.best_model_name,
                'total_crops': len(recommender.label_encoder.classes_),
                'accuracy': round(
                    recommender.models[
                        recommender.best_model_name
                    ]['accuracy'],
                    4
                ),
                'features': recommender.feature_names
            },
            'input_parameters': soil_data
        }

        return response

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error in crop recommendation: {e}")
        raise


if __name__ == "__main__":
    # Test the system
    print("Testing Crop Recommendation System...")

    test_params = SoilParameters(
        pH=7.0, N=120, P=54, K=100,
        Fe=4.06, Mn=1.68, Zn=0.83, Cu=0.46, B=0.3,
        Water=500, Moisture=60, Temperature=28, Rainfall=600,
        top_n=5
    )

    result = recommend_crops(test_params)
    print("\n✓ Recommendations generated successfully!")
    print(f"Number of recommendations: {len(result['recommendations'])}")
    for rec in result['recommendations']:
        score = rec['suitability_score']
        conf = rec['confidence']
        print(f"  {rec['rank']}. {rec['crop_name']} - {score}% ({conf})")
