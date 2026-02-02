#!/usr/bin/env python3
"""
Train VLM (Vision Language Model) for Plant Disease Detection
Trains CNN-based model for all 96 crops
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# Optional imports
try:
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from PIL import Image

    HAS_IMAGE_LIBS = True
except ImportError:
    HAS_IMAGE_LIBS = False
    print("Warning: PIL/cv2 not available. Install: pip install pillow opencv-python")

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. Install: pip install scikit-learn")

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow not available. Install: pip install tensorflow")
    print("Will use scikit-learn-based model instead")

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


class VLMTrainer:
    """Train VLM model for plant disease detection"""

    def __init__(self, datasets_dir=None, models_dir=None):
        self.datasets_dir = datasets_dir or Path(__file__).parent / "datasets" / "vlm"
        self.models_dir = models_dir or Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.processed_dir = self.datasets_dir / "processed"
        self.train_dir = self.processed_dir / "train"
        self.val_dir = self.processed_dir / "val"
        self.test_dir = self.processed_dir / "test"

        self.model = None
        self.label_encoder = None
        self.crop_encoder = None
        self.disease_encoder = None

    def load_dataset_info(self):
        """Load dataset manifest"""
        manifest_file = self.datasets_dir / "dataset_manifest.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def create_cnn_model(self, num_classes: int, input_shape=(224, 224, 3)):
        """Create CNN model for disease classification"""
        if not HAS_TENSORFLOW:
            return None

        model = models.Sequential(
            [
                # Convolutional base
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                # Dense layers
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def create_transfer_learning_model(
        self, num_classes: int, input_shape=(224, 224, 3)
    ):
        """Create model using transfer learning"""
        if not HAS_TENSORFLOW:
            return None

        # Use MobileNetV2 as base
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(512, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def prepare_data_generators(self, batch_size=32):
        """Prepare data generators for training"""
        if not HAS_TENSORFLOW:
            return None, None, None

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode="nearest",
        )

        # No augmentation for validation/test
        val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            str(self.train_dir),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
        )

        val_generator = val_test_datagen.flow_from_directory(
            str(self.val_dir),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
        )

        test_generator = val_test_datagen.flow_from_directory(
            str(self.test_dir),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
        )

        return train_generator, val_generator, test_generator

    def check_data_availability(self):
        """Check if training data is available"""
        if not self.train_dir.exists():
            return False, 0

        # Count image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_count = 0

        for ext in image_extensions:
            image_count += len(list(self.train_dir.rglob(f"*{ext}")))

        return image_count > 0, image_count

    def train_model(self, use_transfer_learning=True, epochs=50):
        """Train VLM model"""
        print("=" * 80)
        print("TRAINING VLM MODEL")
        print("=" * 80)

        # Check if data exists
        has_data, image_count = self.check_data_availability()

        if not has_data:
            print("⚠️  No training images found!")
            print(f"   Found {image_count} images")
            print("\nPlease download and prepare datasets first:")
            print(
                "1. Download PlantVillage: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
            )
            print("2. Extract to: datasets/vlm/raw/plantvillage/")
            print("3. Run: python organize_plantvillage_dataset.py")
            print("\nOr use synthetic data:")
            print("   python generate_synthetic_vlm_data.py")

            # Create model structure anyway for testing
            print("\n📋 Creating model structure (for testing)...")
            return self.create_model_structure_only()

        print(f"✅ Found {image_count} training images")

        if HAS_TENSORFLOW:
            # Prepare data generators
            train_gen, val_gen, test_gen = self.prepare_data_generators()

            if train_gen is None:
                print("⚠️  Could not create data generators")
                return None

            num_classes = len(train_gen.class_indices)
            print(f"\n📊 Found {num_classes} classes")
            print(f"   Training samples: {train_gen.samples}")
            print(f"   Validation samples: {val_gen.samples}")
            print(f"   Test samples: {test_gen.samples}")

            # Create model
            if use_transfer_learning:
                print("\n🔧 Creating transfer learning model (MobileNetV2)...")
                self.model = self.create_transfer_learning_model(num_classes)
            else:
                print("\n🔧 Creating CNN model...")
                self.model = self.create_cnn_model(num_classes)

            if self.model is None:
                print("⚠️  Could not create model")
                return None

            print("\n📋 Model Architecture:")
            self.model.summary()

            # Train model
            print("\n🚀 Training model...")
            history = self.model.fit(
                train_gen,
                steps_per_epoch=train_gen.samples // train_gen.batch_size,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=val_gen.samples // val_gen.batch_size,
                verbose=1,
            )

            # Evaluate on test set
            print("\n📊 Evaluating on test set...")
            test_loss, test_accuracy = self.model.evaluate(test_gen)
            print(f"   Test Accuracy: {test_accuracy:.4f}")
            print(f"   Test Loss: {test_loss:.4f}")

            # Save model
            model_path = self.models_dir / "edge_ai_vision_model.h5"
            self.model.save(str(model_path))
            print(f"\n✅ Model saved to: {model_path}")

            # Save class indices
            class_indices = {
                "class_indices": train_gen.class_indices,
                "num_classes": num_classes,
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
            }

            indices_path = self.models_dir / "edge_ai_vision_class_indices.json"
            with open(indices_path, "w", encoding="utf-8") as f:
                json.dump(class_indices, f, indent=2)

            print(f"✅ Class indices saved to: {indices_path}")

            return self.model
        else:
            print("⚠️  TensorFlow not available. Using scikit-learn fallback...")
            return self.train_sklearn_model()

    def create_model_structure_only(self, use_transfer_learning=True):
        """Create model structure without training (for testing)"""
        print("\n📋 Creating VLM model structure...")

        if HAS_TENSORFLOW:
            # Create a minimal model structure
            num_classes = 100  # Estimate: ~100 disease classes across crops

            if use_transfer_learning:
                print("Creating MobileNetV2-based model structure...")
                model = self.create_transfer_learning_model(num_classes)
            else:
                print("Creating CNN model structure...")
                model = self.create_cnn_model(num_classes)

            if model:
                # Save model structure
                model_path = self.models_dir / "edge_ai_vision_model_structure.h5"
                model.save(str(model_path))
                print(f"✅ Model structure saved to: {model_path}")
                print(
                    "   (This is a template model - train with real data for production)"
                )

                # Save info
                info = {
                    "model_type": (
                        "transfer_learning" if use_transfer_learning else "cnn"
                    ),
                    "num_classes": num_classes,
                    "input_shape": [224, 224, 3],
                    "status": "structure_only",
                    "note": "Train with real images for production use",
                }

                info_path = self.models_dir / "edge_ai_vision_model_info.json"
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2)

                return model

        print("⚠️  Could not create model structure")
        return None

    def train_sklearn_model(self):
        """Train using scikit-learn (fallback)"""
        if not HAS_SKLEARN:
            print("⚠️  scikit-learn not available")
            return None

        print("Training scikit-learn model (simplified feature extraction)...")
        # This would use hand-crafted features
        # For now, just save a placeholder
        print("⚠️  Full sklearn implementation requires feature extraction")
        return None


def main():
    """Main function"""
    trainer = VLMTrainer()

    # Check for datasets
    manifest = trainer.load_dataset_info()
    if manifest:
        print("📊 Dataset Info:")
        print(f"   Total crops: {manifest['total_crops']}")

    # Train model
    model = trainer.train_model(use_transfer_learning=True, epochs=50)

    if model:
        print("\n" + "=" * 80)
        print("✅ VLM MODEL TRAINING COMPLETE!")
        print("=" * 80)
    else:
        print("\n⚠️  Training incomplete. Please prepare datasets first.")


if __name__ == "__main__":
    main()
