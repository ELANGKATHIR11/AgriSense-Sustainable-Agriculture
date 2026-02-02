#!/usr/bin/env python3
"""
Alternative PlantVillage Dataset Downloader
Downloads from GitHub or creates synthetic data structure
"""

import json
import subprocess
from pathlib import Path


def download_from_github():
    """Download PlantVillage from GitHub"""
    print("📥 Attempting to download from GitHub...")

    datasets_dir = Path(__file__).parent.parent / "datasets" / "vlm" / "raw"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    plantvillage_dir = datasets_dir / "plantvillage"

    # Try git clone
    try:
        if not plantvillage_dir.exists():
            print("Cloning PlantVillage repository...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/spMohanty/PlantVillage-Dataset.git",
                    str(plantvillage_dir),
                ],
                check=True,
                timeout=300,
            )
            print("✅ Downloaded from GitHub")
            return True
        else:
            print("✅ PlantVillage directory already exists")
            return True
    except Exception as e:
        print(f"⚠️  Git clone failed: {e}")
        return False


def create_synthetic_structure():
    """Create synthetic dataset structure for training"""
    print("\n📁 Creating synthetic dataset structure...")

    datasets_dir = Path(__file__).parent.parent / "datasets" / "vlm"
    processed_dir = datasets_dir / "processed"

    # Load crop-disease mappings from knowledge base
    knowledge_base = (
        Path(__file__).parent / "knowledge_base" / "disease_knowledge.json"
    )

    if knowledge_base.exists():
        with open(knowledge_base, "r", encoding="utf-8") as f:
            disease_db = json.load(f)
    else:
        disease_db = {}

    # Create structure for all crops
    crops_with_data = [
        "Apple",
        "Maize",
        "Grapes",
        "Tomato",
        "Potato",
        "Chilli",
        "Soybean",
        "Pumpkin",
        "Strawberry",
    ]

    for crop in crops_with_data:
        crop_dir = crop.replace(" ", "_")

        # Get diseases for this crop
        diseases = []
        for disease_name, disease_info in disease_db.items():
            affected_crops = disease_info.get("affected_crops", [])
            if crop in affected_crops:
                diseases.append(disease_name.replace("_", " "))

        if not diseases:
            diseases = ["Leaf Spot", "Root Rot", "Wilt"]

        # Create directories
        for split in ["train", "val", "test"]:
            for disease in diseases + ["Healthy"]:
                disease_dir = (
                    processed_dir
                    / split
                    / crop_dir
                    / disease.replace(" ", "_")
                )
                disease_dir.mkdir(parents=True, exist_ok=True)

                # Create placeholder file
                placeholder = disease_dir / ".gitkeep"
                placeholder.touch()

        print(f"✅ Created structure for {crop}")

    print(f"\n✅ Created synthetic structure for {len(crops_with_data)} crops")
    print("Note: Place actual images in these directories for training")
    return True


def main():
    """Main function"""
    print("=" * 80)
    print("ALTERNATIVE PLANTVILLAGE DOWNLOADER")
    print("=" * 80)

    # Try GitHub download
    success = download_from_github()

    if not success:
        print(
            "\n⚠️  Direct download failed. Creating structure for manual download..."
        )
        create_synthetic_structure()
        print("\n📋 Manual Download Instructions:")
        print("1. Visit: https://github.com/spMohanty/PlantVillage-Dataset")
        print("2. Download ZIP or clone repository")
        print("3. Extract to: backend/ml/datasets/vlm/raw/plantvillage/")
        print("4. Run: python organize_plantvillage_dataset.py")
    else:
        print("\n✅ Download successful!")
        print("Next: Run organize_plantvillage_dataset.py")


if __name__ == "__main__":
    main()
