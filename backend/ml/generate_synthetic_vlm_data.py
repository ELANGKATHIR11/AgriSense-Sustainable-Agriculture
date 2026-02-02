#!/usr/bin/env python3
"""
Generate Synthetic VLM Training Data
Creates augmented images and prepares dataset for training
"""

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Optional imports
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image, ImageEnhance, ImageFilter

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

HAS_IMAGE_LIBS = HAS_PIL or HAS_CV2

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


def create_synthetic_image(width=224, height=224, crop_type="leaf"):
    """Create a synthetic plant image"""
    if not HAS_IMAGE_LIBS:
        return None

    # Create base image (green background for leaves)
    if crop_type == "leaf":
        base_color = (34, 139, 34)  # Forest green
    else:
        base_color = (255, 255, 255)  # White

    img = Image.new("RGB", (width, height), base_color)

    # Add some texture/noise
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.9 + np.random.random() * 0.2)

    return img


def augment_image(image, num_augmentations=5):
    """Apply data augmentation to an image"""
    if not HAS_IMAGE_LIBS or image is None:
        return []

    augmented = []

    for i in range(num_augmentations):
        aug_img = image.copy()

        # Random rotation
        angle = np.random.uniform(-15, 15)
        aug_img = aug_img.rotate(angle, fillcolor=(255, 255, 255))

        # Random flip
        if np.random.random() > 0.5:
            aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)

        # Brightness adjustment
        enhancer = ImageEnhance.Brightness(aug_img)
        aug_img = enhancer.enhance(0.8 + np.random.random() * 0.4)

        # Contrast adjustment
        enhancer = ImageEnhance.Contrast(aug_img)
        aug_img = enhancer.enhance(0.8 + np.random.random() * 0.4)

        # Color adjustment
        enhancer = ImageEnhance.Color(aug_img)
        aug_img = enhancer.enhance(0.9 + np.random.random() * 0.2)

        # Add slight blur
        if np.random.random() > 0.7:
            aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=1))

        augmented.append(aug_img)

    return augmented


def generate_dataset_structure():
    """Generate complete dataset structure with synthetic data"""
    print("=" * 80)
    print("GENERATING SYNTHETIC VLM DATASET")
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

    # Crops with potential real data (from PlantVillage)
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

    total_images = 0

    for crop in crops_with_data:
        crop_dir = crop.replace(" ", "_")

        # Get diseases for this crop
        diseases = []
        for disease_name, disease_info in disease_db.items():
            affected_crops = disease_info.get("affected_crops", [])
            if crop in affected_crops:
                diseases.append(disease_name.replace("_", " "))

        if not diseases:
            diseases = ["Leaf Spot", "Root Rot", "Wilt", "Powdery Mildew"]

        # Create synthetic images for each disease
        for split, num_images in [("train", 50), ("val", 10), ("test", 10)]:
            for disease in diseases + ["Healthy"]:
                disease_dir = (
                    processed_dir
                    / split
                    / crop_dir
                    / disease.replace(" ", "_")
                )
                disease_dir.mkdir(parents=True, exist_ok=True)

                # Create base synthetic image
                base_img = create_synthetic_image()

                if base_img and HAS_IMAGE_LIBS:
                    # Save base image
                    base_img.save(
                        disease_dir
                        / f'{crop_dir}_{disease.replace(" ", "_")}_000.jpg'
                    )
                    total_images += 1

                    # Generate augmented versions
                    if split == "train":  # More augmentation for training
                        augmented = augment_image(
                            base_img, num_augmentations=num_images - 1
                        )
                        for idx, aug_img in enumerate(augmented):
                            aug_img.save(
                                disease_dir
                                / f'{crop_dir}_{disease.replace(" ", "_")}_{idx+1:03d}.jpg'
                            )
                            total_images += 1
                    else:
                        # Fewer for val/test
                        augmented = augment_image(
                            base_img, num_augmentations=num_images - 1
                        )
                        for idx, aug_img in enumerate(
                            augmented[: num_images - 1]
                        ):
                            aug_img.save(
                                disease_dir
                                / f'{crop_dir}_{disease.replace(" ", "_")}_{idx+1:03d}.jpg'
                            )
                            total_images += 1
                else:
                    # Create placeholder files
                    placeholder = disease_dir / ".gitkeep"
                    placeholder.touch()

        print(f"✅ Generated structure for {crop}")

    print(f"\n✅ Generated {total_images} synthetic images")
    print(f"📁 Dataset structure created in: {processed_dir}")

    return total_images


def main():
    """Main function"""
    if not HAS_IMAGE_LIBS:
        print("⚠️  Image libraries not available. Creating structure only...")
        print("Install: pip install pillow opencv-python")

    total = generate_dataset_structure()

    print("\n" + "=" * 80)
    print("✅ SYNTHETIC DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated {total} synthetic images")
    print("\nNext steps:")
    print(
        "1. Replace synthetic images with real PlantVillage images when available"
    )
    print("2. Run: python train_vlm_model.py")
    print(
        "\nNote: Synthetic data is for structure/testing. Real images will improve accuracy."
    )


if __name__ == "__main__":
    main()
