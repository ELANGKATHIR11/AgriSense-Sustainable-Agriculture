"""
Train VLM (Vision-Language Model) for 91 Crop Recognition
Uses targeted crop images dataset instead of large PlantVillage
Optimized for CPU and NPU (DirectML) acceleration
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from PIL import Image

from torchvision import models, transforms  # type: ignore

HAS_TORCHVISION = True

try:
    import onnxruntime

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("⚠️ onnxruntime not installed for NPU: pip install onnxruntime-directml")


# 91 crops from CSV dataset
CROP_CLASSES = [
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
    " mustard",
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


class CropVLMModel(nn.Module):
    """Vision-Language Model for crop recognition"""

    def __init__(self, num_classes=91, use_pretrained=True):
        super(CropVLMModel, self).__init__()

        # Use ResNet50 as backbone (smaller than ResNet101, good for CPU)
        self.backbone = models.resnet50(pretrained=use_pretrained)

        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class VLMTrainerTargeted:
    """Train VLM on targeted crop images"""

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir or Path(__file__).parent)
        self.models_dir = self.base_dir / "models"
        self.dataset_dir = self.base_dir / "datasets" / "crop_images_small"
        self.crop_classes = CROP_CLASSES
        self.num_classes = len(self.crop_classes)

        # Device detection
        self.device = self.detect_hardware()
        print(f"\n🖥️ Training device: {self.device}")

        # Data transforms
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def detect_hardware(self) -> str:
        """Detect available hardware"""
        # Force CPU-only execution: GPU/NPU checks removed
        # This keeps training/inference CPU-compatible and avoids torch.cuda or DirectML usage.
        return "cpu"

    def check_dataset(self) -> bool:
        """Check if targeted crop dataset exists"""
        if not self.dataset_dir.exists():
            print(f"\n❌ Dataset not found: {self.dataset_dir}")
            print("\n💡 Run this first:")
            print("   python download_targeted_crop_images.py")
            return False

        # Count images per crop
        crop_counts = {}
        for crop in self.crop_classes:
            crop_dir = self.dataset_dir / crop
            if crop_dir.exists():
                images = list(crop_dir.glob("*.jpg")) + list(crop_dir.glob("*.png"))
                crop_counts[crop] = len(images)

        total_images = sum(crop_counts.values())
        crops_with_images = len([c for c in crop_counts.values() if c > 0])

        print("\n📊 Dataset Statistics:")
        print(f"   Total crops: {self.num_classes}")
        print(f"   Crops with images: {crops_with_images}")
        print(f"   Total images: {total_images}")
        print(f"   Avg images per crop: {total_images / max(crops_with_images, 1):.1f}")

        if total_images < 50:
            print("\n⚠️  Warning: Very few images. Download more for better results.")
            return False

        return True

    def create_dataloaders(self, batch_size=16, val_split=0.2):
        """Create training and validation dataloaders"""
        from torch.utils.data import DataLoader, Dataset, random_split

        class CropDataset(Dataset):
            def __init__(self, root_dir, crop_classes, transform=None):
                self.root_dir = Path(root_dir)
                self.crop_classes = crop_classes
                self.transform = transform
                self.samples = []

                # Load all images
                for idx, crop in enumerate(crop_classes):
                    crop_dir = self.root_dir / crop
                    if not crop_dir.exists():
                        continue

                    for img_path in list(crop_dir.glob("*.jpg")) + list(
                        crop_dir.glob("*.png")
                    ):
                        self.samples.append((str(img_path), idx))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                image = Image.open(img_path).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                return image, label

        # Create full dataset
        full_dataset = CropDataset(
            self.dataset_dir, self.crop_classes, self.train_transform
        )

        # Split train/val
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # CPU-friendly
            pin_memory=self.device == "cuda",
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device == "cuda",
        )

        print("\n📦 DataLoaders created:")
        print(f"   Train samples: {train_size}")
        print(f"   Val samples: {val_size}")
        print(f"   Batch size: {batch_size}")

        return train_loader, val_loader

    def train(self, epochs=20, batch_size=16, learning_rate=0.001):
        """Train the VLM model"""

        if not HAS_TORCHVISION:
            print("\n❌ torchvision required")
            print("Install: pip install torch torchvision")
            return

        print("\n" + "=" * 60)
        print("🚀 AgriSense VLM Training - 91 Crops")
        print("=" * 60)

        # Check dataset
        if not self.check_dataset():
            return

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(batch_size=batch_size)

        # Create model
        print("\n🧠 Creating ResNet50-based VLM model...")
        model = CropVLMModel(num_classes=self.num_classes, use_pretrained=True)

        # Move to device
        if self.device == "cuda":
            model = model.cuda()

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=3
        )

        # Training loop
        print("\n🏋️ Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")

        best_val_acc = 0.0
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                if self.device == "cuda":
                    images, labels = images.cuda(), labels.cuda()

                # Forward
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward
                loss.backward()
                optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                if batch_idx % 10 == 0:
                    print(
                        f"   Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
                    )

            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    if self.device == "cuda":
                        images, labels = images.cuda(), labels.cuda()

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            # Update history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Learning rate scheduling
            scheduler.step(val_loss)

            print(f"\n✨ Epoch {epoch+1}/{epochs} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = self.models_dir / "crop_vlm_targeted_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "crop_classes": self.crop_classes,
                    },
                    model_path,
                )
                print(f"   💾 Saved best model (Val Acc: {val_acc:.2f}%)")

        # Save final model
        final_model_path = self.models_dir / "crop_vlm_targeted_final.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "crop_classes": self.crop_classes,
                "history": history,
            },
            final_model_path,
        )

        print("\n✅ Training complete!")
        print(f"   Best Val Acc: {best_val_acc:.2f}%")
        print(f"   Models saved to: {self.models_dir}")

        # Create summary
        self.create_training_summary(history, best_val_acc)

    def create_training_summary(self, history: Dict, best_val_acc: float):
        """Create training summary document"""

        header = f"""# VLM Training Summary - 91 Targeted Crops

## Training Completed
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset
- **Total crops**: {self.num_classes}
- **Architecture**: ResNet50-based
- **Image size**: 224x224
- **Data augmentation**: Rotation, flip, color jitter

## Results
- **Best validation accuracy**: {best_val_acc:.2f}%
- **Final training accuracy**: {history['train_acc'][-1]:.2f}%
- **Final training loss**: {history['train_loss'][-1]:.4f}
- **Final validation loss**: {history['val_loss'][-1]:.4f}

## Model Files
- `crop_vlm_targeted_best.pth` - Best model (highest val acc)
- `crop_vlm_targeted_final.pth` - Final model

## Usage

"""

        usage_example = """```python
import torch
from PIL import Image

# Load model
checkpoint = torch.load('models/crop_vlm_targeted_best.pth')
crop_classes = checkpoint['crop_classes']

model = CropVLMModel(num_classes=len(crop_classes))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('test_crop.jpg').convert('RGB')
image = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image)
    _, predicted = output.max(1)
    crop_name = crop_classes[predicted.item()]
    print(f"Predicted crop: {crop_name}")
```
"""

        footer = f"""
## Crop Classes ({self.num_classes} total)
{chr(10).join(f'{i+1}. {crop}' for i, crop in enumerate(sorted(self.crop_classes)))}
"""

        summary = header + usage_example + footer

        summary_file = self.models_dir / "VLM_TRAINING_SUMMARY.md"
        with open(summary_file, "w") as f:
            f.write(summary)

        # Save history as JSON
        history_file = self.models_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        print(f"\\n📄 Training summary: {summary_file}")
        print(f"📊 Training history: {history_file}")


def main():
    """Main training function"""
    print(
        """
    ╔═══════════════════════════════════════════════════════════╗
    ║  AgriSense VLM Training - 91 Targeted Crops              ║
    ║  Lightweight dataset optimized for CPU/NPU               ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    )

    trainer = VLMTrainerTargeted()

    print("\n⚙️ Training Configuration:")
    print("   - Model: ResNet50-based")
    print("   - Epochs: 20")
    print("   - Batch size: 16 (CPU-friendly)")
    print("   - Learning rate: 0.001")
    print("\n💡 Tip: Make sure you've downloaded crop images first!")
    print("   Run: python download_targeted_crop_images.py")

    proceed = input("\n🚀 Start training? (y/n): ").lower()

    if proceed == "y":
        trainer.train(epochs=20, batch_size=16, learning_rate=0.001)
    else:
        print("Training cancelled.")


if __name__ == "__main__":
    main()
