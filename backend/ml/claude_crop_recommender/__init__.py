"""
Claude Crop Recommendation System
Intelligent agricultural decision support for 100+ Indian crops
"""

__version__ = '1.0.0'
__author__ = 'AgriSense Team'

from .crop_recommendation_ml_model import CropRecommendationSystem
from .crop_recommender_api import recommend_crops, SoilParameters

__all__ = [
    'CropRecommendationSystem',
    'recommend_crops',
    'SoilParameters'
]
