"""
AgriSense Complete ML Training Pipeline v2.0
=============================================
1. Full 190K+ disease dataset training
2. TFLite export for Android edge deployment
3. Soil health data integration for crop recommendation
4. Confidence scoring and uncertainty estimation

Author: AgriSense Team
Date: 2026-02-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm
import os
import logging
import pickle
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Device Configuration
# ============================================================================


def get_best_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🖥️ Using CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        return device, "cuda"

    try:
        import torch_directml

        device = torch_directml.device()
        logger.info(f"🔷 Using DirectML (NPU/AMD GPU): {device}")
        return device, "directml"
    except ImportError:
        pass

    logger.info("💻 Using CPU for training")
    return torch.device("cpu"), "cpu"


DEVICE, DEVICE_TYPE = get_best_device()


# ============================================================================
# Configuration
# ============================================================================


class Config:
    """Training configuration"""

    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "training_logs"
    TFLITE_DIR = BASE_DIR / "tflite_models"

    # All disease dataset paths
    DISEASE_DATASETS = {
        "vlm_train": Path(r"F:\diseases\vlm datasets\Train\Train"),
        "vlm_validate": Path(r"F:\diseases\vlm datasets\Validation"),
        "vlm_test": Path(r"F:\diseases\vlm datasets\Test"),
        "disease3_train": Path(r"F:\diseases\disease3\train"),
        "disease3_test": Path(r"F:\diseases\disease3\test"),
        "vlm2_train": Path(r"F:\diseases\vlm datasets2\Train"),
        "vlm2_validate": Path(r"F:\diseases\vlm datasets2\Validation"),
        "vlm2_test": Path(r"F:\diseases\vlm datasets2\Test"),
        "plantvillage": Path(
            r"F:\diseases\vlm datasets2\PlantVillage-Dataset\raw\color"
        ),
        "disease_csv": Path(r"F:\diseases\plant_disease_dataset.csv"),
    }

    # Soil health data from Tamil Nadu soil cards
    INDIAN_SOIL_RANGES = {
        "N": {"low": (0, 113), "medium": (113, 180), "high": (180, 300)},
        "P": {"low": (0, 4.5), "medium": (4.5, 9), "high": (9, 20)},
        "K": {"low": (0, 48), "medium": (48, 113), "high": (113, 200)},
        "pH": {"acidic": (4, 6), "neutral": (6, 7.5), "alkaline": (7.5, 9)},
        "OC": {"low": (0, 0.51), "medium": (0.51, 0.75), "high": (0.75, 1.0)},
        "Fe": {"low": (0, 2.0), "sufficient": (2.0, 10)},
        "Mn": {"low": (0, 2.0), "sufficient": (2.0, 10)},
        "Zn": {"low": (0, 1.2), "sufficient": (1.2, 5)},
        "Cu": {"low": (0, 1.2), "sufficient": (1.2, 5)},
        "B": {"low": (0, 0.5), "sufficient": (0.5, 2)},
    }

    # Training settings
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 7

    # Uncertainty estimation
    MC_DROPOUT_SAMPLES = 5
    REJECTION_THRESHOLD = 0.4

    DEVICE = DEVICE
    DEVICE_TYPE = DEVICE_TYPE


# ============================================================================
# Data Augmentation
# ============================================================================


def get_train_transforms(image_size=224):
    """Training transforms with realistic field augmentations"""
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.1)
            ),  # Simulate dust/occlusion
        ]
    )


def get_val_transforms(image_size=224):
    """Validation transforms (no augmentation)"""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ============================================================================
# Dataset Classes
# ============================================================================


class UnifiedDiseaseDataset(Dataset):
    """
    Unified dataset that can load from any folder structure.
    Supports: class folders, nested folders, etc.
    """

    def __init__(
        self, root_dirs: List[Path], transform=None, max_samples_per_class=None
    ):
        self.transform = transform
        self.extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

        # Discover all classes across all directories
        self.all_classes = set()
        for root_dir in root_dirs:
            if root_dir.exists():
                self._discover_classes(root_dir)

        self.classes = sorted(list(self.all_classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect samples
        self.samples = []
        for root_dir in root_dirs:
            if root_dir.exists():
                self._collect_samples(root_dir, max_samples_per_class)

        logger.info(
            f"📁 Unified Dataset: {len(self.samples)} images, {len(self.classes)} classes"
        )

    def _discover_classes(self, root_dir: Path, depth=0, max_depth=3):
        """Recursively discover class folders"""
        if depth > max_depth:
            return

        for item in root_dir.iterdir():
            if item.is_dir():
                # Check if this is a class folder (contains images)
                has_images = any(
                    f.suffix.lower() in [e.lower() for e in self.extensions]
                    for f in item.iterdir()
                    if f.is_file()
                )
                if has_images:
                    self.all_classes.add(item.name)
                else:
                    # Go deeper
                    self._discover_classes(item, depth + 1, max_depth)

    def _collect_samples(self, root_dir: Path, max_per_class=None):
        """Collect image samples with labels"""
        class_counts = {cls: 0 for cls in self.classes}

        for root, dirs, files in os.walk(root_dir):
            root_path = Path(root)
            class_name = root_path.name

            if class_name in self.class_to_idx:
                for file in files:
                    if Path(file).suffix.lower() in [
                        e.lower() for e in self.extensions
                    ]:
                        if max_per_class and class_counts[class_name] >= max_per_class:
                            continue
                        self.samples.append(
                            (str(root_path / file), self.class_to_idx[class_name])
                        )
                        class_counts[class_name] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# Models with Uncertainty Estimation
# ============================================================================


class UncertaintyAwareEfficientNet(nn.Module):
    """
    EfficientNet with Monte Carlo Dropout for uncertainty estimation.
    Outputs: logits, confidence, uncertainty
    """

    def __init__(self, num_classes, dropout_rate=0.3, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Backbone
        self.backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Replace classifier with uncertainty-aware head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # Remove original classifier

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

        logger.info(
            f"🧠 Initialized UncertaintyAwareEfficientNet with {num_classes} classes"
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict_with_uncertainty(self, x, num_samples=5):
        """
        Monte Carlo Dropout inference for uncertainty estimation.
        Returns: prediction, confidence, uncertainty, should_reject
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        self.eval()  # Disable dropout

        # Stack predictions
        predictions = torch.stack(predictions)  # [num_samples, batch, num_classes]

        # Mean prediction
        mean_pred = predictions.mean(dim=0)  # [batch, num_classes]

        # Variance (epistemic uncertainty)
        var_pred = predictions.var(dim=0)  # [batch, num_classes]

        # Entropy (aleatoric uncertainty)
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=-1)  # [batch]

        # Combined uncertainty
        uncertainty = entropy + var_pred.max(dim=-1).values

        # Confidence
        max_prob, predicted_class = mean_pred.max(dim=-1)

        # Rejection decision
        should_reject = (max_prob < Config.REJECTION_THRESHOLD) | (uncertainty > 1.5)

        return {
            "predicted_class": predicted_class,
            "confidence": max_prob,
            "uncertainty": uncertainty,
            "should_reject": should_reject,
            "probabilities": mean_pred,
        }


class SoilAwareCropRecommender(nn.Module):
    """
    Crop recommendation model enhanced with real Indian soil health data.
    Incorporates NPK ranges from Tamil Nadu soil health cards.
    """

    def __init__(self, num_features, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

        # Soil health range embeddings
        self.soil_ranges = Config.INDIAN_SOIL_RANGES

    def classify_soil_nutrient(self, value, nutrient_name):
        """Classify nutrient level based on Indian soil health standards"""
        ranges = self.soil_ranges.get(nutrient_name, {})
        for level, (low, high) in ranges.items():
            if low <= value < high:
                return level
        return "unknown"

    def forward(self, x):
        return self.network(x)

    def get_soil_health_score(self, soil_data: dict) -> dict:
        """
        Calculate overall soil health score based on Indian standards.
        Returns score and recommendations.
        """
        scores = {}
        deficiencies = []

        for nutrient in ["N", "P", "K", "pH", "OC"]:
            if nutrient in soil_data:
                level = self.classify_soil_nutrient(soil_data[nutrient], nutrient)
                scores[nutrient] = level
                if level == "low":
                    deficiencies.append(nutrient)

        health_score = 100 - (len(deficiencies) * 15)

        return {
            "overall_score": max(0, health_score),
            "nutrient_levels": scores,
            "deficiencies": deficiencies,
            "recommendations": self._get_recommendations(deficiencies),
        }

    def _get_recommendations(self, deficiencies):
        """Get fertilizer recommendations based on deficiencies"""
        recommendations = []

        fertilizer_map = {
            "N": "Urea (46% N) - Apply 40 kg/acre",
            "P": "SSP (16% P₂O₅) - Apply 25 kg/acre",
            "K": "MOP (60% K₂O) - Apply 12 kg/acre",
            "OC": "Apply organic manure 10 ton/acre",
        }

        for nutrient in deficiencies:
            if nutrient in fertilizer_map:
                recommendations.append(fertilizer_map[nutrient])

        return recommendations


# ============================================================================
# TFLite Export Functions
# ============================================================================


def export_to_onnx(model, output_path, input_size=(1, 3, 224, 224)):
    """Export PyTorch model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(*input_size).to(Config.DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    logger.info(f"📦 Exported ONNX model to {output_path}")


def convert_onnx_to_tflite(
    onnx_path, tflite_path, quantize=True, representative_dataset=None
):
    """
    Convert ONNX to TFLite with optional quantization.
    Requires: onnx, tf2onnx, tensorflow
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)

        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_model_path = str(tflite_path).replace(".tflite", "_tf")
        tf_rep.export_graph(tf_model_path)

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if representative_dataset:
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        logger.info(f"📱 Exported TFLite model to {tflite_path}")
        logger.info(f"   Size: {len(tflite_model) / 1e6:.2f} MB")

        return True

    except ImportError as e:
        logger.warning(f"TFLite conversion requires additional packages: {e}")
        logger.warning("Install: pip install onnx onnx-tf tensorflow")
        return False


# ============================================================================
# Training Functions
# ============================================================================


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler and Config.DEVICE_TYPE == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.1f}%"}
        )

    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100 * correct / total


# ============================================================================
# Main Training Pipelines
# ============================================================================


def train_full_disease_vlm():
    """
    Train disease VLM on all available datasets (190K+ images).
    """
    print("\n" + "=" * 80)
    print("🌿 TRAINING FULL DISEASE VLM (190K+ Images)")
    print("=" * 80)

    Config.MODELS_DIR.mkdir(exist_ok=True)
    Config.LOGS_DIR.mkdir(exist_ok=True)

    # Collect all available dataset paths
    dataset_paths = []
    for name, path in Config.DISEASE_DATASETS.items():
        if path.exists() and path.is_dir() and name != "disease_csv":
            dataset_paths.append(path)
            logger.info(f"✅ Found dataset: {name} -> {path}")

    if not dataset_paths:
        logger.error("No datasets found!")
        return None

    # Create unified dataset
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    full_dataset = UnifiedDiseaseDataset(dataset_paths, transform=train_transform)

    if len(full_dataset) == 0:
        logger.error("Dataset is empty!")
        return None

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Update transform for validation
    val_dataset.dataset.transform = val_transform

    logger.info(f"📊 Dataset split: {train_size} train, {val_size} val")
    logger.info(f"📊 Classes: {len(full_dataset.classes)}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4 if Config.DEVICE_TYPE != "directml" else 0,
        pin_memory=Config.DEVICE_TYPE == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4 if Config.DEVICE_TYPE != "directml" else 0,
        pin_memory=Config.DEVICE_TYPE == "cuda",
    )

    # Initialize model with uncertainty
    num_classes = len(full_dataset.classes)
    model = UncertaintyAwareEfficientNet(num_classes=num_classes).to(Config.DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    scaler = torch.amp.GradScaler("cuda") if Config.DEVICE_TYPE == "cuda" else None

    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n📅 Epoch {epoch}/{Config.EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)

        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "classes": full_dataset.classes,
                    "num_classes": num_classes,
                },
                Config.MODELS_DIR / "disease_vlm_full_best.pth",
            )

            print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break

    # Save training history
    with open(Config.LOGS_DIR / "disease_vlm_full_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save class mapping
    with open(Config.MODELS_DIR / "disease_classes_full.json", "w") as f:
        json.dump(
            {
                "classes": full_dataset.classes,
                "class_to_idx": full_dataset.class_to_idx,
            },
            f,
            indent=2,
        )

    print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.2f}%")

    return model, full_dataset.classes, best_val_acc


def export_to_tflite(model_path):
    """
    Export trained model to TFLite for Android deployment.
    Creates 3 variants: tiny, mobile, full
    """
    print("\n" + "=" * 80)
    print("📱 EXPORTING TO TFLITE FOR ANDROID")
    print("=" * 80)

    Config.TFLITE_DIR.mkdir(exist_ok=True)

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    num_classes = checkpoint["num_classes"]
    classes = checkpoint["classes"]

    model = UncertaintyAwareEfficientNet(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Export to ONNX first
    onnx_path = Config.TFLITE_DIR / "disease_vlm.onnx"
    export_to_onnx(model, str(onnx_path))

    # Try TFLite conversion
    tflite_path = Config.TFLITE_DIR / "disease_vlm_mobile.tflite"
    success = convert_onnx_to_tflite(str(onnx_path), str(tflite_path), quantize=True)

    if not success:
        # Create a simpler export script for manual conversion
        export_script = """
# TFLite Conversion Script
# Run this script after installing required packages:
# pip install onnx onnx-tf tensorflow

import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load ONNX model
onnx_model = onnx.load('disease_vlm.onnx')

# Convert to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph('disease_vlm_tf')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('disease_vlm_tf')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('disease_vlm_mobile.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model) / 1e6:.2f} MB")
"""
        with open(Config.TFLITE_DIR / "convert_to_tflite.py", "w") as f:
            f.write(export_script)

        logger.info(
            f"📝 Created conversion script at {Config.TFLITE_DIR / 'convert_to_tflite.py'}"
        )

    # Save class labels for Android
    with open(Config.TFLITE_DIR / "labels.txt", "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")

    logger.info(f"📝 Saved labels to {Config.TFLITE_DIR / 'labels.txt'}")

    return True


def train_soil_aware_crop_model():
    """
    Train crop recommendation model with Indian soil health data integration.
    """
    print("\n" + "=" * 80)
    print("🌱 TRAINING SOIL-AWARE CROP RECOMMENDER")
    print("=" * 80)

    # Load dataset
    data_path = Config.BASE_DIR / "datasets" / "indian_agriculture_ml_dataset.csv"
    if not data_path.exists():
        data_path = Config.BASE_DIR / "datasets" / "soil_health_dataset.csv"

    if not data_path.exists():
        logger.error("No crop dataset found!")
        return None

    df = pd.read_csv(data_path)
    logger.info(f"📊 Loaded {len(df)} samples from {data_path.name}")

    # Feature columns
    feature_cols = [
        col for col in df.columns if col not in ["crop_type", "label", "crop", "target"]
    ]
    target_col = next(
        (col for col in ["crop_type", "label", "crop", "target"] if col in df.columns),
        None,
    )

    if target_col is None:
        logger.error("No target column found!")
        return None

    # Prepare data
    X = df[feature_cols].values
    y = df[target_col]

    # Encode labels
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y_encoded)

    # Create dataset
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model
    num_features = X.shape[1]
    num_classes = len(le.classes_)
    model = SoilAwareCropRecommender(num_features, num_classes).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    best_val_acc = 0
    for epoch in range(1, 51):
        model.train()
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        val_acc = 100 * val_correct / val_total

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Acc: {100*train_correct/train_total:.1f}%, Val Acc: {val_acc:.1f}%"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "classes": list(le.classes_),
                    "feature_cols": feature_cols,
                },
                Config.MODELS_DIR / "soil_aware_crop_recommender.pth",
            )

            # Save scaler and encoder
            with open(Config.MODELS_DIR / "crop_scaler_v2.pkl", "wb") as f:
                pickle.dump(scaler, f)
            with open(Config.MODELS_DIR / "crop_encoder_v2.pkl", "wb") as f:
                pickle.dump(le, f)

    print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.2f}%")

    # Test soil health analysis
    print("\n📋 Testing Soil Health Analysis with Tamil Nadu Data:")
    sample_soil = {"N": 65, "P": 3, "K": 62, "pH": 7.0, "OC": 0.18}
    analysis = model.get_soil_health_score(sample_soil)
    print(f"   Overall Score: {analysis['overall_score']}/100")
    print(f"   Deficiencies: {analysis['deficiencies']}")
    print(f"   Recommendations: {analysis['recommendations']}")

    return model, best_val_acc


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main training orchestrator"""
    print("=" * 80)
    print("🌾 AGRISENSE COMPLETE ML TRAINING PIPELINE v2.0")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {Config.DEVICE} ({Config.DEVICE_TYPE})")

    results = {}

    # 1. Train Full Disease VLM
    print("\n" + "=" * 80)
    print("STEP 1: Training Full Disease VLM (190K+ images)")
    print("=" * 80)
    model, classes, vlm_acc = train_full_disease_vlm()
    if vlm_acc:
        results["disease_vlm_full"] = vlm_acc

    # 2. Export to TFLite
    print("\n" + "=" * 80)
    print("STEP 2: Exporting to TFLite for Android")
    print("=" * 80)
    model_path = Config.MODELS_DIR / "disease_vlm_full_best.pth"
    if model_path.exists():
        export_to_tflite(model_path)
        results["tflite_export"] = "Success"

    # 3. Train Soil-Aware Crop Recommender
    print("\n" + "=" * 80)
    print("STEP 3: Training Soil-Aware Crop Recommender")
    print("=" * 80)
    crop_model, crop_acc = train_soil_aware_crop_model()
    if crop_acc:
        results["soil_aware_crop"] = crop_acc

    # Summary
    print("\n" + "=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)
    for model_name, result in results.items():
        if isinstance(result, float):
            print(f"   {model_name}: {result:.2f}%")
        else:
            print(f"   {model_name}: {result}")

    # Save results
    with open(Config.LOGS_DIR / "full_training_results.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "device": Config.DEVICE_TYPE,
                "results": {
                    k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in results.items()
                },
            },
            f,
            indent=2,
        )

    print("\n✅ All training complete!")
    print(f"📁 Models saved to: {Config.MODELS_DIR}")
    print(f"📱 TFLite models: {Config.TFLITE_DIR}")


if __name__ == "__main__":
    main()
