#!/usr/bin/env python3
"""
Complete VLM Training Setup
Creates structure and prepares for training even without downloaded datasets
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


def setup_complete_structure():
    """Set up complete VLM training structure"""
    print("=" * 80)
    print("SETTING UP VLM TRAINING STRUCTURE")
    print("=" * 80)

    datasets_dir = Path(__file__).parent / "datasets" / "vlm"
    processed_dir = datasets_dir / "processed"

    # Load disease knowledge
    knowledge_base = (
        Path(__file__).parent / "knowledge_base" / "disease_knowledge.json"
    )
    disease_db = {}
    if knowledge_base.exists():
        with open(knowledge_base, "r", encoding="utf-8") as f:
            disease_db = json.load(f)

    # Create structure for all crops
    structure_info = {}

    for crop in ALL_CROPS:
        crop_dir = crop.replace(" ", "_")

        # Get diseases for this crop
        diseases = []
        for disease_name, disease_info in disease_db.items():
            affected_crops = disease_info.get("affected_crops", [])
            if crop in affected_crops:
                diseases.append(disease_name.replace("_", " "))

        if not diseases:
            diseases = ["Leaf Spot", "Root Rot", "Wilt"]

        structure_info[crop] = {
            "diseases": diseases,
            "total_diseases": len(diseases) + 1,  # +1 for Healthy
        }

        # Create directories
        for split in ["train", "val", "test"]:
            for disease in diseases + ["Healthy"]:
                disease_dir_path = (
                    processed_dir
                    / split
                    / crop_dir
                    / disease.replace(" ", "_")
                )
                disease_dir_path.mkdir(parents=True, exist_ok=True)

                # Create README in each directory
                readme = disease_dir_path / "README.txt"
                with open(readme, "w", encoding="utf-8") as f:
                    f.write(f"Place {crop} {disease} images here.\n")
                    f.write("Supported formats: .jpg, .jpeg, .png\n")
                    f.write("Recommended size: 224x224 pixels\n")

    print(f"✅ Created structure for {len(ALL_CROPS)} crops")

    # Save structure info
    info_file = datasets_dir / "training_structure.json"
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(structure_info, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved structure info to: {info_file}")

    return structure_info


def create_training_config():
    """Create training configuration file"""
    config = {
        "model": {
            "type": "transfer_learning",
            "base_model": "MobileNetV2",
            "input_size": [224, 224, 3],
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.0001,
        },
        "data": {
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "augmentation": {
                "rotation_range": 20,
                "width_shift_range": 0.2,
                "height_shift_range": 0.2,
                "horizontal_flip": True,
                "zoom_range": 0.2,
            },
        },
        "paths": {
            "train": "datasets/vlm/processed/train",
            "val": "datasets/vlm/processed/val",
            "test": "datasets/vlm/processed/test",
            "model_output": "models/edge_ai_vision_model.h5",
        },
    }

    config_file = (
        Path(__file__).parent / "datasets" / "vlm" / "training_config.json"
    )
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Created training config: {config_file}")
    return config


def main():
    """Main function"""
    print("\nSetting up VLM training structure...\n")

    # Create structure
    structure_info = setup_complete_structure()

    # Create config
    config = create_training_config()

    print("\n" + "=" * 80)
    print("✅ VLM TRAINING SETUP COMPLETE!")
    print("=" * 80)

    print("\n📊 Structure Created:")
    print(f"   Total crops: {len(ALL_CROPS)}")
    print(
        f"   Total disease categories: {sum(info['total_diseases'] for info in structure_info.values())}"
    )

    print("\n📁 Directory Structure:")
    print("   Train: datasets/vlm/processed/train/")
    print("   Val:   datasets/vlm/processed/val/")
    print("   Test:  datasets/vlm/processed/test/")

    print("\n📋 Next Steps:")
    print("   1. Download PlantVillage dataset:")
    print(
        "      - Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
    )
    print("      - Or: https://github.com/spMohanty/PlantVillage-Dataset")
    print("      - Extract to: datasets/vlm/raw/plantvillage/")
    print("   2. Organize dataset:")
    print("      python organize_plantvillage_dataset.py")
    print("   3. Train model:")
    print("      python train_vlm_model.py")

    print("\n💡 Note: Structure is ready. Add images to train the model!")
    print()


if __name__ == "__main__":
    main()
