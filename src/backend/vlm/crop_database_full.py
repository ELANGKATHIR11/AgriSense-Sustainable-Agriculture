from typing import Dict, List, Optional
from dataclasses import dataclass, field
from .crop_database import Crop, Disease, Weed, INDIAN_CROPS_DB as EXISTING_DB

# List of all 104 crops
ALL_CROPS_LIST = [
    'almond', 'apple', 'arecanut', 'arhar', 'bajra', 'banana', 'barley', 'barnyard_millet', 'beans', 'beetroot', 
    'bitter_gourd', 'black_pepper', 'bottle_gourd', 'brinjal', 'broccoli', 'buck_wheat', 'buckwheat', 'cabbage', 
    'cardamom', 'carrot', 'cashew', 'castor', 'cauliflower', 'chickpea', 'chilli', 'cluster_bean', 'coconut', 
    'coffee', 'coriander', 'cotton', 'cucumber', 'cumin', 'custard_apple', 'dragon_fruit', 'fenugreek', 'field_pea', 
    'foxtail_millet', 'french_bean', 'garlic', 'ginger', 'grapes', 'green_pea', 'groundnut', 'guava', 'horse_gram', 
    'jackfruit', 'jowar', 'jute', 'kidney_bean', 'kodo_millet', 'large_cardamom', 'lentil', 'lettuce', 'linseed', 
    'litchi', 'little_millet', 'maize', 'mandarin_orange', 'mango', 'masoor', 'moong', 'moth_bean', 'muskmelon', 
    'mustard', 'niger', 'oats', 'okra', 'onion', 'orange', 'papaya', 'passion_fruit', 'pearl_millet', 'peas', 
    'pigeon_pea', 'pineapple', 'pomegranate', 'potato', 'proso_millet', 'pumpkin', 'radish', 'ragi', 'raspberry', 
    'rice', 'ridge_gourd', 'rubber', 'safflower', 'sapota', 'sesame', 'sorghum', 'soybean', 'spinach', 'strawberry', 
    'sugarcane', 'sunflower', 'sweet_potato', 'tea', 'tobacco', 'tomato', 'turmeric', 'turnip', 'urad', 'walnut', 
    'watermelon', 'wheat'
]

def create_generic_crop(name: str) -> Crop:
    return Crop(
        name=name.replace('_', ' ').title(),
        scientific_name="Unknown",
        category="general",
        common_diseases=[
            Disease(
                name="General Disease",
                scientific_name="Various pathogens",
                symptoms=["Discoloration", "Wilting"],
                causes=["Pathogens"],
                treatment=["Consult expert"],
                prevention=["Good agricultural practices"],
                affected_parts=["leaves", "stems"]
            )
        ],
        common_weeds=[
            Weed(
                name="General Weed",
                scientific_name="Various species",
                characteristics=["Competes for resources"],
                control_methods={"mechanical": ["Weeding"]},
                competition_impact="Moderate",
                growth_stage_vulnerability=["all"]
            )
        ],
        growth_stages=["vegetative", "reproductive"],
        optimal_conditions={},
        regional_importance=[]
    )

# Create the full database
INDIAN_CROPS_DB: Dict[str, Crop] = EXISTING_DB.copy()

for crop_name in ALL_CROPS_LIST:
    if crop_name not in INDIAN_CROPS_DB:
        INDIAN_CROPS_DB[crop_name] = create_generic_crop(crop_name)

def get_crop_info(crop_name: str) -> Optional[Crop]:
    """Get crop information by name"""
    crop_key = crop_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return INDIAN_CROPS_DB.get(crop_key)

def list_all_crops() -> List[str]:
    """List all available crops"""
    return [crop.name for crop in INDIAN_CROPS_DB.values()]

def search_crops_by_category(category: str) -> List[Crop]:
    """Search crops by category"""
    return [crop for crop in INDIAN_CROPS_DB.values() if crop.category == category]

def get_diseases_for_crop(crop_name: str) -> List[Disease]:
    """Get all diseases for a specific crop"""
    crop = get_crop_info(crop_name)
    return crop.common_diseases if crop else []

def get_weeds_for_crop(crop_name: str) -> List[Weed]:
    """Get all weeds for a specific crop"""
    crop = get_crop_info(crop_name)
    return crop.common_weeds if crop else []
