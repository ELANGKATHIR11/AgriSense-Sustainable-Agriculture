#!/usr/bin/env python3
"""
Generate comprehensive cultivation guides for all 96 crops
Includes water efficiency, fertilizer optimization, and yield enhancement tips
"""

import json
from pathlib import Path

# All 96 crops
ALL_CROPS = [
    "Almond",
    "Apple",
    "Arecanut",
    "Arhar",
    "Bajra",
    "Banana",
    "Barley",
    "Barnyard_Millet",
    "Beetroot",
    "Bitter_Gourd",
    "Black_Pepper",
    "Bottle_Gourd",
    "Brinjal",
    "Buckwheat",
    "Cabbage",
    "Cardamom",
    "Carrot",
    "Cashew",
    "Castor",
    "Cauliflower",
    "Chickpea",
    "Chilli",
    "Cluster_Bean",
    "Coconut",
    "Coffee",
    "Coriander",
    "Cotton",
    "Cucumber",
    "Cumin",
    "Custard_Apple",
    "Dragon_Fruit",
    "Fenugreek",
    "Field_Pea",
    "Foxtail_Millet",
    "French_Bean",
    "Garlic",
    "Ginger",
    "Grapes",
    "Green_Pea",
    "Groundnut",
    "Guava",
    "Horse_Gram",
    "Jackfruit",
    "Jowar",
    "Jute",
    "Kidney_Bean",
    "Kodo_Millet",
    "Lentil",
    "Lettuce",
    "Linseed",
    "Litchi",
    "Little_Millet",
    "Maize",
    "Mango",
    "Masoor",
    "Moong",
    "Moth_Bean",
    "Muskmelon",
    "Mustard",
    "Niger",
    "Oats",
    "Okra",
    "Onion",
    "Orange",
    "Papaya",
    "Pearl_Millet",
    "Pigeon_Pea",
    "Pineapple",
    "Pomegranate",
    "Potato",
    "Proso_Millet",
    "Pumpkin",
    "Radish",
    "Ragi",
    "Rice",
    "Ridge_Gourd",
    "Rubber",
    "Safflower",
    "Sapota",
    "Sesame",
    "Sorghum",
    "Soybean",
    "Spinach",
    "Strawberry",
    "Sugarcane",
    "Sunflower",
    "Sweet_Potato",
    "Tea",
    "Tobacco",
    "Tomato",
    "Turmeric",
    "Turnip",
    "Urad",
    "Walnut",
    "Watermelon",
    "Wheat",
]


def generate_cultivation_guide(crop_name: str) -> dict:
    """Generate comprehensive cultivation guide for a crop"""

    # Crop-specific data (can be enhanced with real agricultural data)
    crop_data = get_crop_specific_data(crop_name)

    guide = {
        "crop_name": crop_name,
        "scientific_name": crop_data.get("scientific_name", "N/A"),
        "crop_type": crop_data.get("type", "General"),
        "overview": f"Comprehensive cultivation guide for {crop_name} focusing on sustainable practices, water efficiency, and yield optimization.",
        "soil_requirements": {
            "soil_type": crop_data.get(
                "soil_type", "Well-drained, fertile loam"
            ),
            "ph_range": crop_data.get("ph_range", "6.0-7.5"),
            "drainage": "Good drainage essential",
            "organic_matter": "2-3% organic matter recommended",
        },
        "climate_requirements": {
            "temperature": crop_data.get("temperature", "20-30°C"),
            "rainfall": crop_data.get("rainfall", "500-1500mm annually"),
            "humidity": crop_data.get("humidity", "Moderate"),
            "season": crop_data.get("season", "Kharif/Rabi"),
        },
        "planting": {
            "method": crop_data.get(
                "planting_method", "Direct seeding or transplanting"
            ),
            "spacing": crop_data.get("spacing", "As per crop requirements"),
            "seed_rate": crop_data.get("seed_rate", "Varies by variety"),
            "planting_time": crop_data.get("planting_time", "Optimal season"),
            "depth": crop_data.get("planting_depth", "2-5 cm"),
        },
        "water_management": {
            "water_requirement": crop_data.get(
                "water_requirement", "Moderate"
            ),
            "irrigation_method": "Drip irrigation recommended (saves 30-50% water)",
            "irrigation_schedule": "Based on soil moisture and growth stage",
            "water_efficiency_tips": [
                "Use drip irrigation instead of flood irrigation",
                "Apply mulch to reduce evaporation by 30-40%",
                "Water during early morning (6-8 AM) or evening (6-8 PM)",
                "Use soil moisture sensors for precision irrigation",
                "Practice deficit irrigation during non-critical stages",
                "Implement rainwater harvesting",
                "Use raised beds for better drainage and water efficiency",
            ],
            "water_savings": "Can reduce water usage by 30-50% while maintaining yield",
        },
        "fertilizer_management": {
            "npk_ratio": crop_data.get("npk_ratio", "Balanced NPK"),
            "fertilizer_application": "Split application based on growth stages",
            "organic_options": "Farmyard manure, compost, vermicompost",
            "fertilizer_efficiency_tips": [
                "Conduct soil testing before applying fertilizers",
                "Use slow-release fertilizers for better nutrient uptake",
                "Apply fertilizers in split doses: 50% basal, 25% at vegetative, 25% at reproductive",
                "Use organic compost (10-15 tons/ha) to improve soil health",
                "Practice crop rotation with legumes for natural N fixation",
                "Use foliar feeding for micronutrients (Zn, B, Fe)",
                "Apply biofertilizers (Azospirillum, PSB) to reduce chemical fertilizer by 20-30%",
                "Use precision agriculture techniques for targeted application",
            ],
            "fertilizer_reduction": "Can reduce fertilizer usage by 20-40% through optimized management",
        },
        "yield_optimization": {
            "target_yield": crop_data.get(
                "target_yield", "Varies by variety and management"
            ),
            "yield_enhancement_tips": [
                "Use high-yielding, disease-resistant varieties",
                "Optimize plant spacing for maximum productivity",
                "Practice timely sowing/planting",
                "Implement integrated nutrient management (INM)",
                "Use proper irrigation scheduling based on crop growth stages",
                "Control weeds, pests, and diseases timely",
                "Practice crop rotation and intercropping",
                "Use growth regulators and bio-stimulants",
                "Harvest at optimal maturity stage",
                "Post-harvest management for quality preservation",
            ],
            "yield_increase": "Can increase yield by 15-30% through optimized practices",
        },
        "pest_management": {
            "common_pests": crop_data.get("pests", ["Monitor regularly"]),
            "ipm_approach": [
                "Use pest-resistant varieties",
                "Practice crop rotation",
                "Maintain field sanitation",
                "Use biological controls (beneficial insects, neem-based products)",
                "Apply chemical pesticides only when economic threshold is reached",
                "Monitor pest populations regularly",
            ],
        },
        "disease_management": {
            "common_diseases": crop_data.get(
                "diseases", ["Monitor regularly"]
            ),
            "prevention": [
                "Use disease-resistant varieties",
                "Practice crop rotation",
                "Maintain proper spacing for air circulation",
                "Apply preventive fungicides during high-risk periods",
                "Remove and destroy infected plant parts",
                "Ensure proper drainage",
            ],
        },
        "harvesting": {
            "harvest_time": crop_data.get("harvest_time", "At maturity"),
            "harvest_method": crop_data.get(
                "harvest_method", "Manual or mechanical"
            ),
            "post_harvest": "Proper storage and handling for quality preservation",
        },
        "sustainability_tips": [
            "Practice conservation agriculture",
            "Use organic inputs where possible",
            "Implement water-efficient irrigation",
            "Reduce chemical inputs through IPM and INM",
            "Maintain soil health through organic matter addition",
            "Practice crop diversification",
        ],
    }

    return guide


def get_crop_specific_data(crop_name: str) -> dict:
    """Get crop-specific agricultural data"""
    # Enhanced data for common crops, defaults for others
    crop_db = {
        "Rice": {
            "scientific_name": "Oryza sativa",
            "type": "Cereal",
            "soil_type": "Clay loam, alluvial",
            "ph_range": "5.5-6.5",
            "temperature": "20-35°C",
            "rainfall": "1000-2000mm",
            "planting_method": "Transplanting",
            "spacing": "20x15 cm",
            "seed_rate": "20-25 kg/ha",
            "planting_time": "June-July (Kharif)",
            "water_requirement": "High (1200-1500mm)",
            "npk_ratio": "100:50:50 kg/ha",
            "target_yield": "4-6 tons/ha",
            "pests": ["Brown planthopper", "Stem borer", "Leaf folder"],
            "diseases": ["Blast", "Brown spot", "Sheath blight"],
            "harvest_time": "120-150 days",
        },
        "Wheat": {
            "scientific_name": "Triticum aestivum",
            "type": "Cereal",
            "soil_type": "Well-drained loam",
            "ph_range": "6.0-7.5",
            "temperature": "15-25°C",
            "rainfall": "400-600mm",
            "planting_method": "Direct seeding",
            "spacing": "20-25 cm row spacing",
            "seed_rate": "100-125 kg/ha",
            "planting_time": "October-November (Rabi)",
            "water_requirement": "Moderate (400-500mm)",
            "npk_ratio": "120:60:40 kg/ha",
            "target_yield": "4-5 tons/ha",
            "pests": ["Aphids", "Termites", "Army worm"],
            "diseases": ["Rust", "Karnal bunt", "Powdery mildew"],
            "harvest_time": "120-140 days",
        },
        "Maize": {
            "scientific_name": "Zea mays",
            "type": "Cereal",
            "soil_type": "Well-drained loam",
            "ph_range": "6.0-7.5",
            "temperature": "18-27°C",
            "rainfall": "500-800mm",
            "planting_method": "Direct seeding",
            "spacing": "75x25 cm",
            "seed_rate": "20-25 kg/ha",
            "planting_time": "June-July (Kharif)",
            "water_requirement": "Moderate (500-600mm)",
            "npk_ratio": "120:60:40 kg/ha",
            "target_yield": "5-7 tons/ha",
            "pests": ["Fall armyworm", "Stem borer"],
            "diseases": ["Turcicum leaf blight", "Downy mildew"],
            "harvest_time": "80-100 days",
        },
        "Tomato": {
            "scientific_name": "Solanum lycopersicum",
            "type": "Vegetable",
            "soil_type": "Well-drained sandy loam",
            "ph_range": "6.0-7.0",
            "temperature": "20-30°C",
            "rainfall": "600-1000mm",
            "planting_method": "Transplanting",
            "spacing": "60x45 cm",
            "seed_rate": "200-300 g/ha",
            "planting_time": "Year-round (protected)",
            "water_requirement": "Moderate (600-800mm)",
            "npk_ratio": "150:100:100 kg/ha",
            "target_yield": "40-60 tons/ha",
            "pests": ["Aphids", "Whitefly", "Fruit borer"],
            "diseases": ["Early blight", "Late blight", "Bacterial wilt"],
            "harvest_time": "90-120 days",
        },
        "Potato": {
            "scientific_name": "Solanum tuberosum",
            "type": "Tuber",
            "soil_type": "Sandy loam, well-drained",
            "ph_range": "5.5-6.5",
            "temperature": "15-25°C",
            "rainfall": "500-700mm",
            "planting_method": "Tuber planting",
            "spacing": "60x20 cm",
            "seed_rate": "2-2.5 tons/ha",
            "planting_time": "October-November",
            "water_requirement": "Moderate (500-600mm)",
            "npk_ratio": "120:80:120 kg/ha",
            "target_yield": "25-35 tons/ha",
            "pests": ["Aphids", "Cutworms"],
            "diseases": ["Late blight", "Early blight", "Bacterial wilt"],
            "harvest_time": "90-120 days",
        },
    }

    return crop_db.get(
        crop_name,
        {
            "type": "General",
            "soil_type": "Well-drained, fertile",
            "ph_range": "6.0-7.5",
            "temperature": "20-30°C",
            "rainfall": "500-1500mm",
            "planting_method": "Direct seeding or transplanting",
            "water_requirement": "Moderate",
            "npk_ratio": "Balanced NPK",
            "target_yield": "Varies by variety",
        },
    )


def generate_all_guides():
    """Generate guides for all 96 crops"""
    guides = {}

    print(f"Generating cultivation guides for {len(ALL_CROPS)} crops...")

    for crop in ALL_CROPS:
        crop_key = crop.replace(" ", "_")
        guides[crop_key] = generate_cultivation_guide(crop)
        print(f"✅ Generated guide for {crop}")

    return guides


def main():
    """Main function to generate and save all guides"""
    output_dir = Path(__file__).parent / "knowledge_base"
    output_dir.mkdir(exist_ok=True)

    guides = generate_all_guides()

    output_file = output_dir / "cultivation_guides.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(guides, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Generated {len(guides)} cultivation guides")
    print(f"📁 Saved to: {output_file}")


if __name__ == "__main__":
    main()
