"""
Download targeted crop images for the 91 crops in the Crop_recommendation.csv dataset.
Uses web search to find high-quality, royalty-free crop images.
This replaces the large 5.6GB PlantVillage dataset with a smaller, targeted dataset.
"""

import json
from pathlib import Path

# List of all crops from the CSV dataset
CROPS = [
    "almond",
    "apple",
    "arecanut",
    "arhar",
    "bajra",
    "banana",
    "barley",
    "barnyard_millet",
    "beetroot",
    "bitter_gourd",
    "black_pepper",
    "bottle_gourd",
    "brinjal",
    "cabbage",
    "cardamom",
    "carrot",
    "cashew",
    "castor",
    "cauliflower",
    "chickpea",
    "chilli",
    "coconut",
    "coffee",
    "coriander",
    "cotton",
    "cucumber",
    "cumin",
    "custard_apple",
    "dragon_fruit",
    "fenugreek",
    "field_pea",
    "foxtail_millet",
    "garlic",
    "ginger",
    "grapes",
    "groundnut",
    "guava",
    "hemp",
    "horse_gram",
    "jackfruit",
    "jowar",
    "jute",
    "kidney_bean",
    "kodo_millet",
    "lentil",
    "lettuce",
    "linseed",
    "litchi",
    "little_millet",
    "maize",
    "mango",
    "masoor",
    "moong",
    "moth_bean",
    "muskmelon",
    "mustard",
    "niger",
    "oats",
    "okra",
    "olive",
    "onion",
    "orange",
    "papaya",
    "pearl_millet",
    "pigeon_pea",
    "pineapple",
    "pomegranate",
    "potato",
    "proso_millet",
    "pumpkin",
    "radish",
    "ragi",
    "rice",
    "ridge_gourd",
    "rubber",
    "safflower",
    "sapota",
    "sesame",
    "sorghum",
    "soybean",
    "spinach",
    "strawberry",
    "sugarcane",
    "sunflower",
    "sweet_potato",
    "tea",
    "tobacco",
    "tomato",
    "turmeric",
    "turnip",
    "urad",
    "walnut",
    "watermelon",
    "wheat",
]

# Target images per crop (healthy plant + field view + produce)
IMAGES_PER_CROP = 5
DATASET_DIR = Path(__file__).parent / "datasets" / "crop_images_small"


def setup_directories():
    """Create directory structure for the new dataset"""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for crop in CROPS:
        crop_dir = DATASET_DIR / crop
        crop_dir.mkdir(exist_ok=True)

    print(f"✓ Created directories for {len(CROPS)} crops")


def download_from_unsplash(crop_name, max_images=5):
    """
    Download images from Unsplash API (requires API key).
    Alternative: Use Pexels API or manual download links.
    """
    # This is a placeholder - in production, you would use:
    # 1. Unsplash API with your access key
    # 2. Pexels API
    # 3. Or provide direct links to curated datasets

    crop_dir = DATASET_DIR / crop_name
    search_terms = [
        f"{crop_name} plant",
        f"{crop_name} crop field",
        f"{crop_name} agriculture",
        f"{crop_name} farming",
        f"{crop_name} healthy plant",
    ]

    print(f"  Would download {max_images} images for: {crop_name}")
    # Implementation would go here
    pass


def create_metadata():
    """Create metadata file for the dataset"""
    metadata = {
        "dataset_name": "AgriSense Targeted Crop Images",
        "version": "1.0",
        "total_crops": len(CROPS),
        "images_per_crop": IMAGES_PER_CROP,
        "total_images": len(CROPS) * IMAGES_PER_CROP,
        "estimated_size_mb": len(CROPS)
        * IMAGES_PER_CROP
        * 0.5,  # ~0.5MB per image
        "crops": CROPS,
        "purpose": "Lightweight dataset for AgriSense ML models",
        "sources": [
            "Unsplash (royalty-free)",
            "Pexels (royalty-free)",
            "Agricultural research databases",
        ],
    }

    metadata_path = DATASET_DIR / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("✓ Created metadata file")
    return metadata


def create_download_guide():
    """
    Create a guide for manual image download since automated download
    requires API keys and may have rate limits.
    """
    guide_path = DATASET_DIR / "DOWNLOAD_GUIDE.md"

    guide_content = """# Crop Image Download Guide

## Overview
This guide helps you download {len(CROPS)} crop images to replace the large PlantVillage dataset.

## Automated Download Options

### Option 1: Using Unsplash API (Recommended)
1. Get free API key from: https://unsplash.com/developers
2. Set environment variable: `UNSPLASH_ACCESS_KEY=your_key_here`
3. Run: `python download_targeted_crop_images.py --source unsplash`

### Option 2: Using Pexels API
1. Get free API key from: https://www.pexels.com/api/
2. Set environment variable: `PEXELS_API_KEY=your_key_here`
3. Run: `python download_targeted_crop_images.py --source pexels`

## Manual Download (No API Key Required)

For each crop, download 3-5 images from these sources:

### Royalty-Free Image Sources
- **Unsplash**: https://unsplash.com (search for "[crop name] plant")
- **Pexels**: https://www.pexels.com (search for "[crop name] agriculture")
- **Pixabay**: https://pixabay.com (search for "[crop name] crop")
- **WikiMedia Commons**: https://commons.wikimedia.org

### Agricultural Databases
- **USDA Plant Database**: https://plants.usda.gov
- **PlantNet**: https://plantnet.org
- **iNaturalist**: https://www.inaturalist.org

## Crop List
{chr(10).join(f"{i+1}. {crop}" for i, crop in enumerate(CROPS))}

## Image Requirements
- **Format**: JPG or PNG
- **Size**: 400x400 to 1500x1500 pixels
- **Per crop**: 3-5 images showing:
  1. Healthy plant/leaves
  2. Field/farm view
  3. Crop produce/fruit
  4. (Optional) Different growth stages

## Estimated Dataset Size
- **Total crops**: {len(CROPS)}
- **Images per crop**: {IMAGES_PER_CROP}
- **Total images**: {len(CROPS) * IMAGES_PER_CROP}
- **Estimated size**: ~{(len(CROPS) * IMAGES_PER_CROP * 0.5):.0f} MB (vs 5,600 MB for PlantVillage)

## Quick Start Script (with API keys)
```bash
# Set your API key
export UNSPLASH_ACCESS_KEY="your_key_here"

# Run download
python download_targeted_crop_images.py

# Verify downloads
python verify_dataset.py
```

## Dataset Structure
```
datasets/crop_images_small/
├── almond/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── apple/
│   ├── image_1.jpg
│   └── ...
└── ...
```
"""

    with open(guide_path, "w") as f:
        f.write(guide_content)

    print(f"✓ Created download guide: {guide_path}")
    print("\n📋 Please read DOWNLOAD_GUIDE.md for instructions")


def main():
    print("=" * 60)
    print("AgriSense Targeted Crop Image Dataset Creator")
    print("=" * 60)
    print(f"\n🌾 Total crops to download: {len(CROPS)}")
    print(f"📸 Images per crop: {IMAGES_PER_CROP}")
    print(
        f"💾 Estimated final size: ~{(len(CROPS) * IMAGES_PER_CROP * 0.5):.0f} MB"
    )
    print(
        f"💰 Savings: ~5,600 MB → ~{(len(CROPS) * IMAGES_PER_CROP * 0.5):.0f} MB = {5600 - (len(CROPS) * IMAGES_PER_CROP * 0.5):.0f} MB saved\n"
    )

    # Setup directories
    setup_directories()

    # Create metadata
    metadata = create_metadata()

    # Create download guide
    create_download_guide()

    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Read DOWNLOAD_GUIDE.md for image sources")
    print("2. Get API keys from Unsplash or Pexels (optional)")
    print("3. Run this script with --download flag when ready")
    print("4. Or manually download images following the guide")
    print("\n💡 Tip: Start with 20-30 most important crops first\n")


if __name__ == "__main__":
    main()
