"""
Enhanced Disease VLM Training Pipeline
=====================================
Trains plant disease detection model on 192K+ image dataset
Supports: CPU, CUDA GPU, DirectML (NPU/AMD GPU)

Author: AgriSense Team
Date: 2026-02-02
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Device Configuration - CPU, GPU, NPU Support
# ============================================================================


def get_best_device():
    """
    Get the best available device for training.
    Priority: CUDA GPU > DirectML (NPU/AMD) > CPU
    """
    # Check for CUDA GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🖥️ Using CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        return device, "cuda"

    # Check for DirectML (Windows NPU/AMD GPU)
    try:
        import torch_directml

        device = torch_directml.device()
        logger.info(f"🔷 Using DirectML (NPU/AMD GPU): {device}")
        return device, "directml"
    except ImportError:
        logger.warning("DirectML not available")

    # Fallback to CPU
    logger.info("💻 Using CPU for training")
    return torch.device("cpu"), "cpu"


# ============================================================================
# Configuration
# ============================================================================


class Config:
    """Training configuration"""

    # Paths
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "training_logs"

    # Disease dataset paths
    DISEASE_DATASETS = {
        "vlm_train": Path(r"F:\diseases\vlm datasets\Train\Train"),
        "plant_disease_416": Path(
            r"F:\diseases\disease2\PlantDisease416x416\PlantDisease416x416\train"
        ),
        "plant_disease_100": Path(r"F:\diseases\disease2\PlantDiseases100x100"),
        "disease_csv": Path(r"F:\diseases\plant_disease_dataset.csv"),
    }

    # Model settings
    IMAGE_SIZE = 224  # Standard for ResNet/EfficientNet
    NUM_CLASSES = None  # Will be determined from dataset

    # Training hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # Data augmentation
    USE_AUGMENTATION = True

    # Mixed precision training
    USE_MIXED_PRECISION = True

    # Device
    DEVICE, DEVICE_TYPE = get_best_device()


# ============================================================================
# Data Augmentation - Realistic Field Conditions
# ============================================================================


def get_train_transforms(image_size=224):
    """
    Training transforms with realistic field condition augmentations.
    Simulates: phone camera blur, variable lighting, dust, angles
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            # Simulate field lighting conditions
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
            ),
            # Simulate camera quality issues
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3
            ),
            transforms.ToTensor(),
            # ImageNet normalization (to be updated with Indian farm stats)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


class PlantDiseaseDataset(Dataset):
    """
    Dataset for plant disease images organized in class folders.
    Supports: Healthy, Powdery, Rust, and any additional classes.
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        extensions=(".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"),
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions

        # Discover classes from folder structure
        self.classes = sorted(
            [
                d.name
                for d in self.root_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all image paths
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [e.lower() for e in extensions]:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

        logger.info(
            f"📁 Loaded {len(self.samples)} images from {len(self.classes)} classes"
        )
        for cls, idx in self.class_to_idx.items():
            count = sum(1 for s in self.samples if s[1] == idx)
            logger.info(f"   Class '{cls}': {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


class EnvironmentalDiseaseDataset(Dataset):
    """
    Dataset for environmental factors affecting disease presence.
    Uses plant_disease_dataset.csv with temperature, humidity, rainfall, soil_pH.
    """

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # Features: temperature, humidity, rainfall, soil_pH
        self.features = ["temperature", "humidity", "rainfall", "soil_pH"]
        self.X = torch.FloatTensor(df[self.features].values)
        self.y = torch.LongTensor(df["disease_present"].values)

        # Normalize features
        self.mean = self.X.mean(dim=0)
        self.std = self.X.std(dim=0)
        self.X = (self.X - self.mean) / self.std

        logger.info(f"📊 Loaded {len(self)} samples from CSV")
        logger.info(
            f"   Disease present: {self.y.sum().item()} ({100*self.y.float().mean():.1f}%)"
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# Model Architectures
# ============================================================================


class DiseaseVLM(nn.Module):
    """
    Vision model for plant disease detection.
    Uses EfficientNet-B3 backbone with custom classification head.
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # Use EfficientNet-B3 for good accuracy/speed tradeoff
        self.backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

        logger.info(f"🧠 Initialized DiseaseVLM with {num_classes} classes")

    def forward(self, x):
        return self.backbone(x)


class EnvironmentalDiseasePredictor(nn.Module):
    """
    Neural network for predicting disease risk from environmental factors.
    Input: temperature, humidity, rainfall, soil_pH
    Output: Binary disease probability
    """

    def __init__(self, input_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # Binary classification
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# Training Functions
# ============================================================================


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Mixed precision training
        if scaler and Config.USE_MIXED_PRECISION and Config.DEVICE_TYPE == "cuda":
            with torch.cuda.amp.autocast():
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


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
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


def train_disease_vlm():
    """
    Train the plant disease VLM on the image dataset.
    Uses 192K+ images from F:\diseases
    """
    print("\n" + "=" * 80)
    print("🌿 TRAINING PLANT DISEASE VLM")
    print("=" * 80)
    print(f"Device: {Config.DEVICE} ({Config.DEVICE_TYPE})")

    # Create directories
    Config.MODELS_DIR.mkdir(exist_ok=True)
    Config.LOGS_DIR.mkdir(exist_ok=True)

    # Load dataset
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    # Use VLM dataset (smaller but cleaner)
    dataset_path = Config.DISEASE_DATASETS["vlm_train"]
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    full_dataset = PlantDiseaseDataset(dataset_path, transform=train_transform)

    # Set number of classes
    Config.NUM_CLASSES = len(full_dataset.classes)

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Override transforms for validation
    val_dataset.dataset.transform = val_transform

    # Create data loaders
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

    # Initialize model
    model = DiseaseVLM(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if Config.DEVICE_TYPE == "cuda" else None

    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n📅 Epoch {epoch}/{Config.EPOCHS}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, scaler
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)

        # Update scheduler
        scheduler.step(val_acc)

        # Log
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "classes": full_dataset.classes,
                },
                Config.MODELS_DIR / "disease_vlm_best.pth",
            )

            print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break

    # Save training history
    with open(Config.LOGS_DIR / "disease_vlm_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc


def train_environmental_model():
    """
    Train the environmental disease risk predictor.
    Uses plant_disease_dataset.csv
    """
    print("\n" + "=" * 80)
    print("🌡️ TRAINING ENVIRONMENTAL DISEASE PREDICTOR")
    print("=" * 80)
    print(f"Device: {Config.DEVICE} ({Config.DEVICE_TYPE})")

    # Load dataset
    csv_path = Config.DISEASE_DATASETS["disease_csv"]
    if not csv_path.exists():
        logger.error(f"Dataset not found: {csv_path}")
        return

    dataset = EnvironmentalDiseaseDataset(csv_path)

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model
    model = EnvironmentalDiseasePredictor().to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    best_val_acc = 0
    for epoch in range(1, 51):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_acc = 100 * correct / total

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                },
                Config.MODELS_DIR / "environmental_disease_model.pth",
            )

    print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main training orchestrator"""
    print("=" * 80)
    print("🌾 AGRISENSE DISEASE ML TRAINING PIPELINE")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {Config.DEVICE} ({Config.DEVICE_TYPE})")
    print(f"Models Directory: {Config.MODELS_DIR}")

    # Check datasets
    print("\n📂 Checking datasets...")
    for name, path in Config.DISEASE_DATASETS.items():
        exists = "✅" if path.exists() else "❌"
        print(f"   {exists} {name}: {path}")

    results = {}

    # Train Disease VLM
    print("\n" + "=" * 80)
    vlm_acc = train_disease_vlm()
    if vlm_acc:
        results["disease_vlm"] = vlm_acc

    # Train Environmental Model
    print("\n" + "=" * 80)
    env_acc = train_environmental_model()
    if env_acc:
        results["environmental_model"] = env_acc

    # Summary
    print("\n" + "=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)
    for model_name, acc in results.items():
        print(f"   {model_name}: {acc:.2f}%")

    # Save results
    with open(Config.LOGS_DIR / "training_results.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "device": Config.DEVICE_TYPE,
                "results": results,
            },
            f,
            indent=2,
        )

    print("\n✅ All training complete!")


if __name__ == "__main__":
    main()
