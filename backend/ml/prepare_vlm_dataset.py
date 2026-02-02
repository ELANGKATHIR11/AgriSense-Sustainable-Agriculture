"""
Dataset Preparation for Plant Disease VLM Training
Downloads and prepares PlantVillage dataset with metadata
"""

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets" / "plant_disease_vlm"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# PlantVillage dataset info
PLANTVILLAGE_URL = (
    "https://github.com/spMohanty/PlantVillage-Dataset/archive/master.zip"
)

# Disease descriptions for text-image pairing
DISEASE_DESCRIPTIONS = {
    # Tomato diseases
    "Tomato___Bacterial_spot": "A tomato leaf affected by bacterial spot disease showing dark brown spots with yellow halos",
    "Tomato___Early_blight": "A tomato leaf with early blight disease exhibiting concentric ring patterns and yellowing",
    "Tomato___Late_blight": "A tomato leaf showing late blight disease with large brown patches and white mold",
    "Tomato___Leaf_Mold": "A tomato leaf infected with leaf mold showing yellow irregular spots on upper surface",
    "Tomato___Septoria_leaf_spot": "A tomato leaf with septoria leaf spot disease showing small circular spots with dark borders",
    "Tomato___Spider_mites_Two_spotted_spider_mite": "A tomato leaf damaged by spider mites showing yellow stippling and webbing",
    "Tomato___Target_Spot": "A tomato leaf with target spot disease exhibiting concentric ring patterns in lesions",
    "Tomato___Tomato_mosaic_virus": "A tomato leaf infected with mosaic virus showing mottled yellow and green patterns",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "A tomato leaf with yellow leaf curl virus showing upward curling and yellowing",
    "Tomato___healthy": "A healthy tomato leaf with vibrant green color and no disease symptoms",
    # Potato diseases
    "Potato___Early_blight": "A potato leaf with early blight disease showing dark brown lesions with concentric rings",
    "Potato___Late_blight": "A potato leaf infected with late blight displaying large irregular brown lesions",
    "Potato___healthy": "A healthy potato leaf with uniform green color and no pathological signs",
    # Pepper diseases
    "Pepper,_bell___Bacterial_spot": "A bell pepper leaf affected by bacterial spot with dark brown lesions",
    "Pepper,_bell___healthy": "A healthy bell pepper leaf with consistent green coloration",
    # Corn diseases
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": "A corn leaf with gray leaf spot disease showing rectangular gray-brown lesions",
    "Corn_(maize)___Common_rust_": "A corn leaf infected with common rust displaying orange-brown pustules",
    "Corn_(maize)___Northern_Leaf_Blight": "A corn leaf with northern leaf blight showing long gray-green cigar-shaped lesions",
    "Corn_(maize)___healthy": "A healthy corn leaf with bright green color and no disease symptoms",
    # Apple diseases
    "Apple___Apple_scab": "An apple leaf infected with scab disease showing olive-green to brown lesions",
    "Apple___Black_rot": "An apple leaf with black rot disease exhibiting purple-bordered lesions",
    "Apple___Cedar_apple_rust": "An apple leaf affected by cedar apple rust with bright orange lesions",
    "Apple___healthy": "A healthy apple leaf with vibrant green color",
    # Grape diseases
    "Grape___Black_rot": "A grape leaf with black rot disease showing brown irregular lesions",
    "Grape___Esca_(Black_Measles)": "A grape leaf infected with esca disease displaying tiger-stripe patterns",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "A grape leaf with leaf blight showing irregular brown spots",
    "Grape___healthy": "A healthy grape leaf with consistent green coloration",
    # Other crops
    "Cherry_(including_sour)___Powdery_mildew": "A cherry leaf infected with powdery mildew showing white powdery coating",
    "Cherry_(including_sour)___healthy": "A healthy cherry leaf with normal green color",
    "Peach___Bacterial_spot": "A peach leaf with bacterial spot disease showing dark lesions",
    "Peach___healthy": "A healthy peach leaf with vibrant green color",
    "Strawberry___Leaf_scorch": "A strawberry leaf with leaf scorch displaying purple-bordered lesions",
    "Strawberry___healthy": "A healthy strawberry leaf with bright green color",
    "Orange___Haunglongbing_(Citrus_greening)": "An orange leaf infected with citrus greening showing yellow blotchy mottling",
    "Squash___Powdery_mildew": "A squash leaf with powdery mildew showing white powdery fungal growth",
    "Raspberry___healthy": "A healthy raspberry leaf with normal green coloration",
    "Soybean___healthy": "A healthy soybean leaf with dark green color",
    "Blueberry___healthy": "A healthy blueberry leaf with vibrant green color",
}


def download_plantvillage(force_download=False):
    """Download PlantVillage dataset"""
    zip_path = DATA_DIR / "plantvillage.zip"
    extract_path = DATA_DIR / "raw"

    if extract_path.exists() and not force_download:
        print(f"✅ Dataset already exists at {extract_path}")
        return extract_path

    print("📥 Downloading PlantVillage dataset...")
    print(f"   URL: {PLANTVILLAGE_URL}")
    print(f"   Destination: {zip_path}")

    # Download
    response = requests.get(PLANTVILLAGE_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(
        desc="Downloading", total=total_size, unit="B", unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"✅ Dataset downloaded and extracted to {extract_path}")
    return extract_path


def prepare_dataset(train_split=0.8, val_split=0.1):
    """Prepare dataset with train/val/test splits and metadata"""

    print("\n" + "=" * 80)
    print("📊 PREPARING PLANT DISEASE DATASET")
    print("=" * 80)

    # Download if needed
    raw_path = download_plantvillage()

    # Find image directory
    image_dirs = list(raw_path.rglob("*color"))
    if not image_dirs:
        raise FileNotFoundError(
            f"Could not find image directory in {raw_path}"
        )

    image_root = image_dirs[0]
    print(f"\n📁 Image directory: {image_root}")

    # Collect all images
    samples = []
    class_names = sorted([d.name for d in image_root.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"\n🔍 Found {len(class_names)} classes:")
    for idx, name in enumerate(class_names[:10]):
        print(f"   {idx}: {name}")
    if len(class_names) > 10:
        print(f"   ... and {len(class_names) - 10} more")

    # Copy images to organized structure
    organized_dir = DATA_DIR / "images"
    organized_dir.mkdir(exist_ok=True)

    print("\n📋 Organizing images...")
    for class_name in tqdm(class_names, desc="Processing classes"):
        class_dir = image_root / class_name
        class_idx = class_to_idx[class_name]

        for img_path in class_dir.glob("*.JPG"):
            # Copy to organized directory
            new_name = f"{class_name}_{img_path.name}"
            new_path = organized_dir / new_name

            if not new_path.exists():
                shutil.copy(img_path, new_path)

            samples.append(
                {
                    "image_path": new_name,
                    "class_id": class_idx,
                    "class_name": class_name,
                }
            )

    print(f"✅ Total samples: {len(samples)}")

    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(samples)

    train_size = int(len(samples) * train_split)
    val_size = int(len(samples) * val_split)

    train_samples = samples[:train_size]
    val_samples = samples[train_size : train_size + val_size]
    test_samples = samples[train_size + val_size :]

    print("\n📊 Dataset splits:")
    print(
        f"   Train: {len(train_samples)} ({len(train_samples)/len(samples)*100:.1f}%)"
    )
    print(
        f"   Val:   {len(val_samples)} ({len(val_samples)/len(samples)*100:.1f}%)"
    )
    print(
        f"   Test:  {len(test_samples)} ({len(test_samples)/len(samples)*100:.1f}%)"
    )

    # Create metadata files
    print("\n💾 Saving metadata...")

    metadata_template = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "descriptions": DISEASE_DESCRIPTIONS,
        "disease_info": {
            name: {
                "description": DISEASE_DESCRIPTIONS.get(
                    name, f"A plant leaf showing {name}"
                ),
                "treatment": "Consult agricultural expert for treatment recommendations",
            }
            for name in class_names
        },
    }

    # Train metadata
    train_metadata = {**metadata_template, "samples": train_samples}
    with open(DATA_DIR / "train_metadata.json", "w") as f:
        json.dump(train_metadata, f, indent=2)

    # Val metadata
    val_metadata = {**metadata_template, "samples": val_samples}
    with open(DATA_DIR / "val_metadata.json", "w") as f:
        json.dump(val_metadata, f, indent=2)

    # Test metadata
    test_metadata = {**metadata_template, "samples": test_samples}
    with open(DATA_DIR / "test_metadata.json", "w") as f:
        json.dump(test_metadata, f, indent=2)

    print(f"✅ Metadata saved to {DATA_DIR}")

    # Print statistics
    print("\n📊 Dataset Statistics:")
    print(f"   Total images: {len(samples)}")
    print(f"   Number of classes: {len(class_names)}")
    print(
        f"   Average images per class: {len(samples) / len(class_names):.1f}"
    )

    print("\n" + "=" * 80)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Dataset location: {DATA_DIR}")
    print(f"📁 Images: {organized_dir}")
    print("\n🚀 Ready to train! Run:")
    print("   python train_native_vlm.py --mode train")

    return DATA_DIR


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare PlantVillage dataset for VLM training"
    )
    parser.add_argument(
        "--train-split", type=float, default=0.8, help="Train split ratio"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download dataset",
    )
    args = parser.parse_args()

    prepare_dataset(train_split=args.train_split, val_split=args.val_split)
