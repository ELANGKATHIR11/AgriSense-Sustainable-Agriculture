"""
VLM Training Script - Adapted for Local PlantVillage Dataset
Uses the processed dataset in backend/ml/datasets/vlm/processed
Architecture: ResNet50 with transfer learning
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np


# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Paths - Updated to use local dataset
    BASE_DIR = Path(__file__).parent
    DATASET_ROOT = BASE_DIR / "datasets" / "vlm" / "processed"
    OUTPUT_DIR = BASE_DIR / "models"

    # Training
    BATCH_SIZE = 16  # Conservative for CPU/moderate GPU
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 7

    # Model
    IMG_SIZE = 224
    NUM_WORKERS = 2  # Reduced for stability

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Dataset Class
# ============================================================================
class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []

        # Scan directory structure
        for class_idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                self.class_names.append(class_dir.name)
                for img_file in class_dir.glob("*.jpg"):
                    self.image_paths.append(str(img_file))
                    self.labels.append(class_idx)
                for img_file in class_dir.glob("*.JPG"):
                    self.image_paths.append(str(img_file))
                    self.labels.append(class_idx)
                for img_file in class_dir.glob("*.png"):
                    self.image_paths.append(str(img_file))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# Model Architecture
# ============================================================================
def create_model(num_classes):
    """Create ResNet50 model with custom classifier"""
    print(f"🏗️ Building ResNet50 for {num_classes} classes...")

    # Load pretrained ResNet50
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)

    # Freeze early layers (transfer learning)
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


# ============================================================================
# Training Loop
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            {
                "loss": f"{running_loss/len(train_loader):.3f}",
                "acc": f"{100.*correct/total:.2f}%",
            }
        )

    return running_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100.0 * correct / total


# ============================================================================
# Main Training Function
# ============================================================================
def train_vlm():
    print("=" * 80)
    print("🌾 VLM TRAINING - PLANT DISEASE DETECTION")
    print("=" * 80)
    print(f"Device: {Config.DEVICE}")
    print(f"Dataset: {Config.DATASET_ROOT}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print()

    # Check dataset exists
    if not Config.DATASET_ROOT.exists():
        print(f"❌ Dataset not found at {Config.DATASET_ROOT}")
        return

    # Data transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = PlantDiseaseDataset(
        Config.DATASET_ROOT / "train", transform=train_transform
    )
    val_dataset = PlantDiseaseDataset(
        Config.DATASET_ROOT / "val", transform=val_transform
    )

    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    print(f"📊 Classes: {len(train_dataset.class_names)}")

    # Save class mapping
    class_mapping = {
        "classes": train_dataset.class_names,
        "num_classes": len(train_dataset.class_names),
    }

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(Config.OUTPUT_DIR / "vlm_disease_classes.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"💾 Saved class mapping\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == "cuda" else False,
    )

    # Create model
    model = create_model(len(train_dataset.class_names))
    model = model.to(Config.DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0

    print("🚀 Starting training...\n")

    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": train_dataset.class_names,
            }

            save_path = Config.OUTPUT_DIR / "vlm_plant_disease_best.pth"
            torch.save(checkpoint, save_path)
            print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            break

        print()

    print("=" * 80)
    print(f"🎉 Training Complete! Best Val Acc: {best_val_acc:.2f}%")
    print(f"📦 Model saved to: {Config.OUTPUT_DIR / 'vlm_plant_disease_best.pth'}")
    print("=" * 80)


if __name__ == "__main__":
    train_vlm()
