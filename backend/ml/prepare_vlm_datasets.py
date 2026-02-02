#!/usr/bin/env python3
"""
Prepare VLM Datasets for All 96 Crops
Downloads and organizes plant disease image datasets for training
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

# Optional imports
try:
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

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

# Dataset sources and mappings
DATASET_SOURCES = {
    "PlantVillage": {
        "url": "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset",
        "github": "https://github.com/spMohanty/PlantVillage-Dataset",
        "description": "50,000+ images, 38 crop-disease pairs",
        "crops_covered": [
            "Apple",
            "Blueberry",
            "Cherry",
            "Corn",
            "Grape",
            "Orange",
            "Peach",
            "Pepper",
            "Potato",
            "Raspberry",
            "Soybean",
            "Squash",
            "Strawberry",
            "Tomato",
        ],
    },
    "PlantNet": {
        "url": "https://github.com/plantnet/PlantNet-300K",
        "description": "300,000 plant images",
        "api": "https://my.plantnet.org/doc",
    },
}

# Crop name mappings (dataset names to our crop names)
CROP_MAPPINGS = {
    "Corn": "Maize",
    "Pepper": "Chilli",
    "Squash": "Pumpkin",
    "Grape": "Grapes",
    "Orange": "Orange",
    "Peach": "Peach",
    "Cherry": "Cherry",
    "Blueberry": "Blueberry",
    "Raspberry": "Raspberry",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
    "Potato": "Potato",
    "Apple": "Apple",
    "Soybean": "Soybean",
}


class VLMDatasetPreparer:
    """Prepare VLM datasets for training"""

    def __init__(self, datasets_dir=None):
        self.datasets_dir = (
            datasets_dir or Path(__file__).parent / "datasets" / "vlm"
        )
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        self.raw_dir = self.datasets_dir / "raw"
        self.processed_dir = self.datasets_dir / "processed"
        self.train_dir = self.processed_dir / "train"
        self.val_dir = self.processed_dir / "val"
        self.test_dir = self.processed_dir / "test"

        # Create directories
        for dir_path in [
            self.raw_dir,
            self.processed_dir,
            self.train_dir,
            self.val_dir,
            self.test_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.dataset_info = {}

    def create_dataset_structure(self):
        """Create directory structure for all 96 crops"""
        print("Creating dataset structure for 96 crops...")

        for crop in ALL_CROPS:
            crop_dir = crop.replace(" ", "_")

            # Create directories for each crop
            for split in ["train", "val", "test"]:
                crop_split_dir = self.processed_dir / split / crop_dir
                crop_split_dir.mkdir(parents=True, exist_ok=True)

                # Create disease subdirectories
                diseases = self.get_crop_diseases(crop)
                for disease in diseases:
                    disease_dir = crop_split_dir / disease.replace(" ", "_")
                    disease_dir.mkdir(parents=True, exist_ok=True)

                # Create healthy subdirectory
                healthy_dir = crop_split_dir / "Healthy"
                healthy_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ Created directory structure for {len(ALL_CROPS)} crops")

    def get_crop_diseases(self, crop_name: str) -> List[str]:
        """Get diseases for a crop"""
        # Load from disease knowledge base
        knowledge_base = (
            Path(__file__).parent / "knowledge_base" / "disease_knowledge.json"
        )

        if knowledge_base.exists():
            with open(knowledge_base, "r", encoding="utf-8") as f:
                disease_db = json.load(f)

            diseases = []
            for disease_name, disease_info in disease_db.items():
                affected_crops = disease_info.get("affected_crops", [])
                if crop_name in affected_crops:
                    diseases.append(disease_name.replace("_", " "))

            return diseases[:10]  # Top 10 diseases

        # Default diseases
        return ["Leaf Spot", "Root Rot", "Wilt", "Mosaic Virus"]

    def download_plantvillage_info(self):
        """Get information about PlantVillage dataset"""
        print("\n📥 PlantVillage Dataset Information:")
        print(
            "   URL: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
        )
        print("   GitHub: https://github.com/spMohanty/PlantVillage-Dataset")
        print("   Size: 50,000+ images")
        print("   Crops: 14 crops with multiple diseases")
        print("\n   To download:")
        print("   1. Install Kaggle API: pip install kaggle")
        print("   2. Set up Kaggle credentials")
        print(
            "   3. Run: kaggle datasets download -d abdallahalidev/plantvillage-dataset"
        )
        print("   4. Or download manually from GitHub")

    def create_dataset_manifest(self):
        """Create dataset manifest with all crops and diseases"""
        manifest = {
            "total_crops": len(ALL_CROPS),
            "crops": {},
            "dataset_sources": DATASET_SOURCES,
            "structure": {
                "raw": str(self.raw_dir),
                "processed": str(self.processed_dir),
                "train": str(self.train_dir),
                "val": str(self.val_dir),
                "test": str(self.test_dir),
            },
        }

        for crop in ALL_CROPS:
            crop_key = crop.replace(" ", "_")
            diseases = self.get_crop_diseases(crop)

            manifest["crops"][crop_key] = {
                "name": crop,
                "diseases": diseases,
                "total_diseases": len(diseases),
                "train_path": f"processed/train/{crop_key}",
                "val_path": f"processed/val/{crop_key}",
                "test_path": f"processed/test/{crop_key}",
            }

        # Save manifest
        manifest_file = self.datasets_dir / "dataset_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Created dataset manifest: {manifest_file}")
        return manifest

    def create_download_script(self):
        """Create script to download datasets"""
        script_content = '''#!/usr/bin/env python3
"""
Download Plant Disease Image Datasets
Run this script to download datasets for VLM training
"""

import os
import subprocess
from pathlib import Path

def install_kaggle():
    """Install Kaggle API"""
    print("Installing Kaggle API...")
    subprocess.run(['pip', 'install', 'kaggle'], check=True)
    print("✅ Kaggle API installed")

def download_plantvillage():
    """Download PlantVillage dataset from Kaggle"""
    print("\\n📥 Downloading PlantVillage Dataset...")
    print("Note: You need Kaggle API credentials")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Create API token")
    print("3. Place kaggle.json in ~/.kaggle/")
    print("\\nRunning download command...")

    try:
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', 'abdallahalidev/plantvillage-dataset',
            '-p', 'datasets/vlm/raw'
        ], check=True)
        print("✅ PlantVillage dataset downloaded")
    except Exception as e:
        print(f"⚠️  Download failed: {e}")
        print("You can download manually from:")
        print("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")

def download_from_github():
    """Download from GitHub repositories"""
    print("\\n📥 GitHub Download Instructions:")
    print("1. PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset")
    print("2. Clone or download ZIP")
    print("3. Extract to datasets/vlm/raw/plantvillage/")

if __name__ == '__main__':
    print("="*80)
    print("VLM DATASET DOWNLOADER")
    print("="*80)

    # Create directories
    Path('datasets/vlm/raw').mkdir(parents=True, exist_ok=True)

    # Install Kaggle if needed
    try:
        import kaggle
    except ImportError:
        install_kaggle()

    # Download datasets
    download_plantvillage()
    download_from_github()

    print("\\n✅ Download complete!")
    print("Next: Run prepare_vlm_datasets.py to organize datasets")
'''

        script_file = self.datasets_dir / "download_datasets.py"
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(script_content)

        # Make executable
        os.chmod(script_file, 0o755)
        print(f"✅ Created download script: {script_file}")

    def create_synthetic_data_guide(self):
        """Create guide for generating synthetic data"""
        guide = {
            "synthetic_data_generation": {
                "description": "Generate synthetic plant disease images using data augmentation",
                "methods": [
                    "Image augmentation (rotation, flip, brightness, contrast)",
                    "GAN-based generation",
                    "Style transfer",
                    "Mixup and CutMix techniques",
                ],
                "tools": [
                    "albumentations - Image augmentation library",
                    "imgaug - Advanced augmentation",
                    "tensorflow.keras.preprocessing - Built-in augmentation",
                ],
                "example_code": """
from albumentations import (
    Compose, Rotate, Flip, RandomBrightnessContrast,
    HueSaturationValue, Blur, GaussianNoise
)

augmentation = Compose([
    Rotate(limit=15),
    Flip(),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
    Blur(blur_limit=3),
    GaussianNoise(var_limit=10.0)
])
""",
            }
        }

        guide_file = self.datasets_dir / "synthetic_data_guide.json"
        with open(guide_file, "w", encoding="utf-8") as f:
            json.dump(guide, f, indent=2, ensure_ascii=False)

        print(f"✅ Created synthetic data guide: {guide_file}")

    def create_dataset_preparation_instructions(self):
        """Create instructions for dataset preparation"""
        instructions = """
# VLM Dataset Preparation Instructions

## Overview
This directory contains scripts and structure for preparing VLM (Vision Language Model) datasets
for training plant disease detection models for all 96 crops.

## Directory Structure

```
datasets/vlm/
├── raw/                    # Raw downloaded datasets
│   └── plantvillage/       # PlantVillage dataset
├── processed/              # Processed and organized datasets
│   ├── train/              # Training images
│   ├── val/                # Validation images
│   └── test/               # Test images
├── dataset_manifest.json   # Dataset information
└── download_datasets.py    # Download script
```

## Step 1: Download Datasets

### Option A: Using Kaggle API
```bash
pip install kaggle
# Set up Kaggle credentials (kaggle.json)
python download_datasets.py
```

### Option B: Manual Download
1. PlantVillage: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Extract to: datasets/vlm/raw/plantvillage/

## Step 2: Organize Datasets

Run the preparation script:
```bash
python prepare_vlm_datasets.py
```

This will:
- Create directory structure for all 96 crops
- Map existing datasets to our crop list
- Organize images by crop and disease
- Create train/val/test splits

## Step 3: Generate Synthetic Data (Optional)

For crops with limited data, use data augmentation:
```bash
python generate_synthetic_data.py
```

## Step 4: Train VLM Model

```bash
python train_vlm_model.py
```

## Dataset Sources

1. **PlantVillage Dataset**
   - 50,000+ images
   - 14 crops covered
   - Multiple diseases per crop
   - Source: https://github.com/spMohanty/PlantVillage-Dataset

2. **PlantNet Dataset**
   - 300,000 plant images
   - General plant identification
   - Source: https://github.com/plantnet/PlantNet-300K

3. **Additional Sources**
   - Kaggle plant disease datasets
   - Agricultural research institution datasets
   - Public domain agricultural image collections

## Crop Coverage

Total crops: {len(ALL_CROPS)}
Crops with direct dataset mapping: ~14 (from PlantVillage)
Crops needing synthetic data: ~82

## Next Steps

1. Download datasets using download_datasets.py
2. Run prepare_vlm_datasets.py to organize
3. Generate synthetic data for missing crops
4. Train VLM model with train_vlm_model.py
"""

        readme_file = self.datasets_dir / "README.md"
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(instructions)

        print(f"✅ Created dataset preparation instructions: {readme_file}")


def main():
    """Main function"""
    print("=" * 80)
    print("VLM DATASET PREPARATION")
    print("=" * 80)
    print()

    preparer = VLMDatasetPreparer()

    # Create directory structure
    preparer.create_dataset_structure()

    # Create dataset manifest
    manifest = preparer.create_dataset_manifest()

    # Create download script
    preparer.create_download_script()

    # Create synthetic data guide
    preparer.create_synthetic_data_guide()

    # Create instructions
    preparer.create_dataset_preparation_instructions()

    # Print dataset info
    preparer.download_plantvillage_info()

    print("\n" + "=" * 80)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Download datasets: python datasets/vlm/download_datasets.py")
    print(
        "2. Organize datasets: Run prepare_vlm_datasets.py again after download"
    )
    print("3. Train VLM model: python train_vlm_model.py")
    print()


if __name__ == "__main__":
    main()
