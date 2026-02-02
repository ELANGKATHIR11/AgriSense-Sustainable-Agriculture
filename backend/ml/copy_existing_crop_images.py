"""
Use existing VLM dataset images for crop training
Copies images from AGRISENSEFULL-STACK/datasets/data/images to crop_images_small
"""

import json
import shutil
from pathlib import Path

# Source from existing VLM dataset
SOURCE_DIR = Path(
    "F:/AGRISENSEFULL-STACK/AGRISENSEFULL-STACK/datasets/data/images"
)
TARGET_DIR = Path(__file__).parent / "datasets" / "crop_images_small"

# Crop mapping from vlm_pairs.csv
CROP_IMAGES = {
    "rice": "Rice.jpg",
    "wheat": "Wheat.jpg",
    "sugarcane": "Sugarcane.jpg",
    "cotton": "Cotton.jpg",
    "jute": "Jute.jpg",
    "groundnut": "Groundnut.jpg",
    "mustard": "Rapeseed_Mustard.jpg",
    "chickpea": "Gram.jpg",
    "arhar": "Tur_Arhar.jpg",
    "maize": "Maize.jpg",
    "jowar": "Jowar.jpg",
    "bajra": "Bajra.jpg",
    "sesame": "Sesamum.jpg",
    "sunflower": "Sunflower.jpg",
    "safflower": "Safflower.jpg",
    "linseed": "Linseed.jpg",
    "castor": "Castor.jpg",
    "niger": "Niger.jpg",
    "tobacco": "Tobacco.jpg",
    "barley": "Barley.jpg",
    "oats": "Oats.jpg",
    "ragi": "Ragi.jpg",
    "coconut": " Coconut.jpg",
    "arecanut": "Arecanut.jpg",
    "coffee": "Coffee.jpg",
    "tea": "Tea.jpg",
    "cardamom": "Cardamom.jpg",
    "black_pepper": "Black_Pepper.jpg",
    "turmeric": "Turmeric.jpg",
    "ginger": "Ginger.jpg",
    "coriander": "Coriander.jpg",
    "cumin": "Cumin.jpg",
    "fenugreek": "Fenugreek.jpg",
    "onion": "Onion.jpg",
    "potato": "Potato.jpg",
    "tomato": "Tomato.jpg",
    "brinjal": "Brinjal.jpg",
    "okra": "Okra.jpg",
    "cabbage": "Cabbage.jpg",
    "cauliflower": "Cauliflower.jpg",
    "carrot": "Carrot.jpg",
    "sweet_potato": "Sweet_Potato.jpg",
}


def copy_existing_images():
    """Copy existing VLM images to crop dataset"""

    if not SOURCE_DIR.exists():
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        return False

    print(f"\n📦 Copying images from: {SOURCE_DIR}")
    print(f"   To: {TARGET_DIR}\n")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for crop, image_file in CROP_IMAGES.items():
        source_file = SOURCE_DIR / image_file

        if source_file.exists():
            # Create crop directory
            crop_dir = TARGET_DIR / crop
            crop_dir.mkdir(exist_ok=True)

            # Copy image
            target_file = crop_dir / "image_1.jpg"
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied {crop}: {image_file}")
            copied += 1
        else:
            print(f"⚠️  Missing {crop}: {image_file}")

    print(f"\n✅ Copied {copied} crop images")

    # Create metadata
    metadata = {
        "dataset_name": "AgriSense Crop Images (from VLM dataset)",
        "total_crops": copied,
        "images_per_crop": 1,
        "source": "AGRISENSEFULL-STACK/datasets/data/images",
        "crops": list(CROP_IMAGES.keys()),
    }

    metadata_file = TARGET_DIR / "dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n📝 Created metadata: {metadata_file}")

    return True


if __name__ == "__main__":
    print(
        """
    ╔══════════════════════════════════════════════════════════════╗
    ║  Copy Existing VLM Images for Crop Training                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    )

    success = copy_existing_images()

    if success:
        print("\n🎯 Next step: Run VLM training")
        print("   python train_vlm_targeted.py")
    else:
        print("\n❌ Setup failed. Check source directory.")
