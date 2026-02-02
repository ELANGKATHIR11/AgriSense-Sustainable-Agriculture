#!/usr/bin/env python3
"""
Organize PlantVillage Dataset into our crop structure
Maps PlantVillage crops to our 96-crop system
"""

import shutil
from pathlib import Path

# PlantVillage to our crop mappings
PLANTVILLAGE_MAPPINGS = {
    "Apple": "Apple",
    "Corn": "Maize",
    "Grape": "Grapes",
    "Peach": "Peach",
    "Pepper": "Chilli",
    "Potato": "Potato",
    "Soybean": "Soybean",
    "Squash": "Pumpkin",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
    "Cherry": "Cherry",
    "Blueberry": "Blueberry",
    "Raspberry": "Raspberry",
    "Orange": "Orange",
}


def organize_plantvillage_dataset(plantvillage_dir, output_dir):
    """Organize PlantVillage dataset into our structure"""
    plantvillage_path = Path(plantvillage_dir)
    output_path = Path(output_dir)

    if not plantvillage_path.exists():
        print(f"❌ PlantVillage directory not found: {plantvillage_dir}")
        print("Please download PlantVillage dataset first")
        return False

    print("📁 Organizing PlantVillage dataset...")
    print(f"   Source: {plantvillage_dir}")
    print(f"   Destination: {output_dir}")

    # Create output directories
    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Process PlantVillage structure
    # PlantVillage typically has: crop_disease/ images
    processed_count = 0

    for crop_folder in plantvillage_path.iterdir():
        if not crop_folder.is_dir():
            continue

        crop_name = crop_folder.name

        # Extract crop and disease from folder name (e.g., "Apple___Apple_scab")
        if "___" in crop_name:
            pv_crop, disease = crop_name.split("___", 1)
        elif "_" in crop_name:
            parts = crop_name.split("_")
            pv_crop = parts[0]
            disease = "_".join(parts[1:])
        else:
            pv_crop = crop_name
            disease = "Healthy"

        # Map to our crop name
        our_crop = PLANTVILLAGE_MAPPINGS.get(pv_crop, pv_crop)

        if our_crop not in PLANTVILLAGE_MAPPINGS.values():
            print(f"⚠️  Skipping unmapped crop: {pv_crop}")
            continue

        # Create destination directories
        crop_dir_name = our_crop.replace(" ", "_")
        disease_dir_name = disease.replace(" ", "_")

        # Split images: 70% train, 15% val, 15% test
        images = list(crop_folder.glob("*.jpg")) + list(
            crop_folder.glob("*.png")
        )

        if len(images) == 0:
            continue

        # Shuffle and split
        import random

        random.seed(42)
        random.shuffle(images)

        n_train = int(len(images) * 0.7)
        n_val = int(len(images) * 0.15)

        train_images = images[:n_train]
        val_images = images[n_train : n_train + n_val]
        test_images = images[n_train + n_val :]

        # Copy images
        for split, img_list in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images),
        ]:
            dest_dir = output_path / split / crop_dir_name / disease_dir_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img in img_list:
                dest_file = dest_dir / img.name
                shutil.copy2(img, dest_file)
                processed_count += 1

        print(
            f"✅ Processed {crop_name}: {len(images)} images → {our_crop}/{disease}"
        )

    print(f"\n✅ Organized {processed_count} images")
    return True


def main():
    """Main function"""
    print("=" * 80)
    print("ORGANIZE PLANTVILLAGE DATASET")
    print("=" * 80)

    # Default paths
    script_dir = Path(__file__).parent
    plantvillage_dir = script_dir / "datasets" / "vlm" / "raw" / "plantvillage"
    output_dir = script_dir / "datasets" / "vlm" / "processed"

    # Check if PlantVillage exists
    if not plantvillage_dir.exists():
        print(f"\n⚠️  PlantVillage dataset not found at: {plantvillage_dir}")
        print("\nPlease download PlantVillage dataset:")
        print(
            "1. From Kaggle: kaggle datasets download -d abdallahalidev/plantvillage-dataset"
        )
        print("2. Extract to: datasets/vlm/raw/plantvillage/")
        print(
            "3. Or clone from GitHub: https://github.com/spMohanty/PlantVillage-Dataset"
        )
        return

    # Organize dataset
    success = organize_plantvillage_dataset(plantvillage_dir, output_dir)

    if success:
        print("\n" + "=" * 80)
        print("✅ DATASET ORGANIZATION COMPLETE!")
        print("=" * 80)
        print("\nNext step: Train VLM model")
        print("Run: python train_vlm_model.py")
    else:
        print(
            "\n⚠️  Organization incomplete. Please check paths and dataset structure."
        )


if __name__ == "__main__":
    main()
