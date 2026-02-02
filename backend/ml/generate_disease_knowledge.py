#!/usr/bin/env python3
"""
Generate comprehensive disease knowledge base for all 96 crops
Includes disease name, symptoms, causes, treatment, cure, and prevention
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

# Common diseases across multiple crops
COMMON_DISEASES = {
    "Leaf Spot": {
        "description": "Fungal disease causing circular or irregular spots on leaves",
        "symptoms": [
            "Circular brown/black spots on leaves",
            "Yellowing around spots",
            "Leaf drop in severe cases",
        ],
        "causes": [
            "Fungal pathogens (Cercospora, Alternaria)",
            "High humidity",
            "Poor air circulation",
            "Overhead irrigation",
        ],
        "affected_crops": [
            "Rice",
            "Wheat",
            "Maize",
            "Tomato",
            "Potato",
            "Cotton",
            "Soybean",
            "Groundnut",
        ],
        "chemical_treatment": [
            "Mancozeb 2g/L water, spray every 7-10 days",
            "Copper oxychloride 2kg/ha",
            "Chlorothalonil 1.5g/L water",
        ],
        "organic_treatment": [
            "Neem oil spray (5ml/L water)",
            "Baking soda solution (1 tsp/L water)",
            "Garlic extract spray",
            "Remove and destroy affected leaves",
        ],
        "prevention": [
            "Use disease-resistant varieties",
            "Maintain proper plant spacing",
            "Avoid overhead irrigation",
            "Practice crop rotation",
            "Remove crop debris after harvest",
        ],
    },
    "Powdery Mildew": {
        "description": "Fungal disease appearing as white powdery growth on leaves and stems",
        "symptoms": [
            "White powdery coating on leaves",
            "Leaf curling and distortion",
            "Premature leaf drop",
        ],
        "causes": [
            "Fungal pathogens (Erysiphe, Podosphaera)",
            "High humidity",
            "Moderate temperatures",
            "Poor air circulation",
        ],
        "affected_crops": [
            "Wheat",
            "Barley",
            "Cucumber",
            "Pumpkin",
            "Grapes",
            "Apple",
            "Pea",
            "Mustard",
        ],
        "chemical_treatment": [
            "Sulfur-based fungicides (2g/L water)",
            "Tebuconazole 0.5ml/L water",
            "Propiconazole 1ml/L water",
        ],
        "organic_treatment": [
            "Milk spray (1:9 ratio with water)",
            "Baking soda solution (1 tsp/L water)",
            "Neem oil spray",
            "Potassium bicarbonate spray",
        ],
        "prevention": [
            "Plant resistant varieties",
            "Ensure good air circulation",
            "Avoid dense planting",
            "Water at base, not on leaves",
            "Apply preventive fungicides in high-risk periods",
        ],
    },
    "Rust": {
        "description": "Fungal disease causing rust-colored pustules on leaves and stems",
        "symptoms": [
            "Rust-colored pustules on leaves",
            "Yellowing and premature leaf drop",
            "Reduced yield",
        ],
        "causes": [
            "Rust fungi (Puccinia spp.)",
            "High humidity",
            "Moderate temperatures",
            "Susceptible varieties",
        ],
        "affected_crops": [
            "Wheat",
            "Barley",
            "Oats",
            "Soybean",
            "Groundnut",
            "Sunflower",
            "Coffee",
        ],
        "chemical_treatment": [
            "Propiconazole 1ml/L water",
            "Tebuconazole 0.5ml/L water",
            "Azoxystrobin 0.5ml/L water",
        ],
        "organic_treatment": [
            "Sulfur dust application",
            "Copper-based fungicides",
            "Remove and destroy infected plant parts",
        ],
        "prevention": [
            "Use rust-resistant varieties",
            "Practice crop rotation",
            "Destroy volunteer plants",
            "Apply preventive fungicides",
            "Avoid late planting",
        ],
    },
    "Blight": {
        "description": "Rapid wilting and death of plant tissues",
        "symptoms": [
            "Water-soaked lesions",
            "Rapid tissue death",
            "Wilting",
            "Dark spots on leaves",
        ],
        "causes": [
            "Bacterial or fungal pathogens",
            "High humidity",
            "Warm temperatures",
            "Wet conditions",
        ],
        "affected_crops": ["Tomato", "Potato", "Rice", "Pepper", "Cucumber"],
        "chemical_treatment": [
            "Mancozeb 2g/L water for early blight",
            "Metalaxyl + Mancozeb for late blight",
            "Copper-based fungicides",
            "Apply every 7-10 days",
        ],
        "organic_treatment": [
            "Copper-based organic fungicides",
            "Remove and destroy affected parts",
            "Improve air circulation",
            "Avoid overhead watering",
        ],
        "prevention": [
            "Use blight-resistant varieties",
            "Practice crop rotation",
            "Ensure proper drainage",
            "Avoid working in wet fields",
            "Remove infected plant debris",
        ],
    },
    "Wilt": {
        "description": "Plant wilting due to vascular system blockage",
        "symptoms": [
            "Wilting of leaves",
            "Yellowing",
            "Stem discoloration",
            "Plant death",
        ],
        "causes": [
            "Fungal pathogens (Fusarium, Verticillium)",
            "Bacterial pathogens",
            "Nematodes",
            "Water stress",
        ],
        "affected_crops": [
            "Tomato",
            "Cotton",
            "Banana",
            "Chilli",
            "Brinjal",
            "Okra",
            "Cucumber",
        ],
        "chemical_treatment": [
            "Carbendazim drenching for fungal wilt",
            "Streptomycin for bacterial wilt",
            "Apply at early stages",
        ],
        "organic_treatment": [
            "Trichoderma application",
            "Neem cake application",
            "Crop rotation with non-host crops",
            "Soil solarization",
        ],
        "prevention": [
            "Use wilt-resistant varieties",
            "Practice long crop rotation",
            "Improve soil drainage",
            "Avoid waterlogging",
            "Use disease-free seeds/seedlings",
        ],
    },
    "Root Rot": {
        "description": "Decay of root system leading to plant decline",
        "symptoms": [
            "Yellowing leaves",
            "Stunted growth",
            "Root discoloration",
            "Plant wilting",
        ],
        "causes": [
            "Fungal pathogens (Rhizoctonia, Pythium)",
            "Waterlogging",
            "Poor drainage",
            "Overwatering",
        ],
        "affected_crops": [
            "Rice",
            "Wheat",
            "Soybean",
            "Cotton",
            "Chilli",
            "Tomato",
            "Pepper",
        ],
        "chemical_treatment": [
            "Metalaxyl seed treatment",
            "Carbendazim drenching",
            "Thiophanate-methyl application",
        ],
        "organic_treatment": [
            "Trichoderma application",
            "Improve drainage",
            "Reduce watering",
            "Apply organic matter",
        ],
        "prevention": [
            "Ensure proper drainage",
            "Avoid waterlogging",
            "Use raised beds",
            "Practice crop rotation",
            "Use disease-free seeds",
        ],
    },
    "Mosaic Virus": {
        "description": "Viral disease causing mottled leaf patterns",
        "symptoms": [
            "Mottled/mosaic patterns on leaves",
            "Leaf distortion",
            "Stunted growth",
            "Reduced yield",
        ],
        "causes": [
            "Viral pathogens",
            "Aphid transmission",
            "Whitefly transmission",
            "Mechanical transmission",
        ],
        "affected_crops": [
            "Tomato",
            "Cucumber",
            "Pumpkin",
            "Tobacco",
            "Potato",
            "Pepper",
            "Beans",
        ],
        "chemical_treatment": [
            "No direct chemical cure for viruses",
            "Control insect vectors with insecticides",
            "Imidacloprid for aphids",
            "Acetamiprid for whiteflies",
        ],
        "organic_treatment": [
            "Remove and destroy infected plants",
            "Control insect vectors",
            "Use reflective mulches",
            "Plant barrier crops",
        ],
        "prevention": [
            "Use virus-free seeds/seedlings",
            "Control insect vectors",
            "Remove infected plants immediately",
            "Practice crop rotation",
            "Use resistant varieties",
            "Avoid working in infected fields",
        ],
    },
    "Downy Mildew": {
        "description": "Fungal disease with downy growth on leaf undersides",
        "symptoms": [
            "Yellow spots on upper leaf surface",
            "Downy growth on lower surface",
            "Leaf drop",
        ],
        "causes": [
            "Fungal pathogens (Peronospora, Plasmopara)",
            "High humidity",
            "Cool temperatures",
            "Wet conditions",
        ],
        "affected_crops": [
            "Grapes",
            "Cucumber",
            "Onion",
            "Lettuce",
            "Sunflower",
            "Sorghum",
        ],
        "chemical_treatment": [
            "Metalaxyl + Mancozeb",
            "Fosetyl-Al",
            "Propamocarb",
            "Apply preventively",
        ],
        "organic_treatment": [
            "Copper-based fungicides",
            "Improve air circulation",
            "Reduce humidity",
            "Remove affected leaves",
        ],
        "prevention": [
            "Use resistant varieties",
            "Ensure good air circulation",
            "Avoid overhead irrigation",
            "Practice crop rotation",
            "Apply preventive fungicides",
        ],
    },
}


def get_crop_specific_diseases(crop_name: str) -> list:
    """Get crop-specific diseases"""
    crop_disease_map = {
        "Rice": [
            "Blast",
            "Brown Spot",
            "Sheath Blight",
            "Bacterial Leaf Blight",
            "Tungro Virus",
        ],
        "Wheat": [
            "Rust",
            "Karnal Bunt",
            "Powdery Mildew",
            "Leaf Blight",
            "Loose Smut",
        ],
        "Maize": ["Turcicum Leaf Blight", "Downy Mildew", "Rust", "Ear Rot"],
        "Tomato": [
            "Early Blight",
            "Late Blight",
            "Bacterial Wilt",
            "Mosaic Virus",
            "Fusarium Wilt",
        ],
        "Potato": [
            "Late Blight",
            "Early Blight",
            "Bacterial Wilt",
            "Blackleg",
            "Scab",
        ],
        "Cotton": [
            "Bacterial Blight",
            "Fusarium Wilt",
            "Verticillium Wilt",
            "Root Rot",
        ],
        "Sugarcane": ["Red Rot", "Smut", "Rust", "Leaf Spot"],
        "Groundnut": ["Early Leaf Spot", "Late Leaf Spot", "Rust", "Stem Rot"],
        "Soybean": ["Rust", "Bacterial Blight", "Mosaic Virus", "Root Rot"],
        "Chilli": [
            "Anthracnose",
            "Bacterial Leaf Spot",
            "Viral Diseases",
            "Damping Off",
        ],
        "Brinjal": [
            "Bacterial Wilt",
            "Phomopsis Blight",
            "Little Leaf",
            "Fruit Rot",
        ],
        "Cucumber": [
            "Powdery Mildew",
            "Downy Mildew",
            "Anthracnose",
            "Angular Leaf Spot",
        ],
        "Grapes": [
            "Downy Mildew",
            "Powdery Mildew",
            "Anthracnose",
            "Black Rot",
        ],
        "Apple": ["Scab", "Powdery Mildew", "Fire Blight", "Cedar Apple Rust"],
        "Banana": ["Panama Disease", "Sigatoka", "Bunchy Top", "Mosaic Virus"],
        "Mango": [
            "Anthracnose",
            "Powdery Mildew",
            "Bacterial Canker",
            "Stem End Rot",
        ],
        "Coffee": ["Coffee Rust", "Berry Disease", "Root Rot"],
        "Tea": ["Blister Blight", "Red Rust", "Brown Blight", "Grey Blight"],
    }

    return crop_disease_map.get(crop_name, ["Leaf Spot", "Root Rot", "Wilt"])


def generate_disease_entry(disease_name: str, crop_name: str = None) -> dict:
    """Generate disease entry"""
    # Check if it's a common disease
    if disease_name in COMMON_DISEASES:
        entry = COMMON_DISEASES[disease_name].copy()
        if crop_name and crop_name not in entry["affected_crops"]:
            entry["affected_crops"].append(crop_name)
        return entry

    # Generate crop-specific disease entry
    return {
        "description": f'{disease_name} affecting {crop_name or "plants"}',
        "symptoms": [
            "Leaf discoloration and spots",
            "Reduced growth and yield",
            "Plant wilting in severe cases",
        ],
        "causes": [
            "Pathogenic infection",
            "Environmental stress",
            "Poor crop management",
        ],
        "affected_crops": [crop_name] if crop_name else [],
        "chemical_treatment": [
            "Apply appropriate fungicide/bactericide",
            "Follow recommended dosage",
            "Repeat application as needed",
        ],
        "organic_treatment": [
            "Remove and destroy affected parts",
            "Apply organic fungicides",
            "Improve crop management practices",
        ],
        "prevention": [
            "Use disease-resistant varieties",
            "Practice crop rotation",
            "Maintain field hygiene",
            "Monitor regularly",
        ],
    }


def generate_all_disease_knowledge():
    """Generate disease knowledge for all crops"""
    disease_knowledge = {}

    # Add common diseases
    disease_knowledge.update(COMMON_DISEASES)

    # Add crop-specific diseases
    for crop in ALL_CROPS:
        crop_diseases = get_crop_specific_diseases(crop)
        for disease_name in crop_diseases:
            disease_key = disease_name.replace(" ", "_").lower()
            if disease_key not in disease_knowledge:
                disease_knowledge[disease_key] = generate_disease_entry(
                    disease_name, crop
                )
            else:
                # Add crop to affected crops list
                if (
                    crop
                    not in disease_knowledge[disease_key]["affected_crops"]
                ):
                    disease_knowledge[disease_key]["affected_crops"].append(
                        crop
                    )

    return disease_knowledge


def main():
    """Main function to generate and save disease knowledge"""
    output_dir = Path(__file__).parent / "knowledge_base"
    output_dir.mkdir(exist_ok=True)

    print("Generating disease knowledge base...")
    disease_knowledge = generate_all_disease_knowledge()

    output_file = output_dir / "disease_knowledge.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(disease_knowledge, f, indent=2, ensure_ascii=False)

    print(
        f"\n✅ Generated disease knowledge for {len(disease_knowledge)} diseases"
    )
    print(f"📁 Saved to: {output_file}")

    # Print summary
    total_affected_crops = sum(
        len(d.get("affected_crops", [])) for d in disease_knowledge.values()
    )
    print(f"📊 Total crop-disease associations: {total_affected_crops}")


if __name__ == "__main__":
    main()
