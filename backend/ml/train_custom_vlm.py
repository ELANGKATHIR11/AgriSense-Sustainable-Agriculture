import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from pathlib import Path
import json
import time
from datetime import datetime

# --- Configuration ---
DATA_DIR = r"F:\vlm datasets\Train\Train"  # Custom Dataset Path
MODELS_DIR = Path("backend/ml/models_custom")
BATCH_SIZE = 16
EPOCHS = 5  # Start with a few epochs to verify
LEARNING_RATE = 0.001

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_custom_model():
    print(f"🚀 Starting Custom VLM Training on: {DATA_DIR}")

    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Training on: {device}")

    # 2. Data Transforms
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        # We don't have a separate Val dir in the leaf folder, so we might split or just use Train for now.
        # Given the folder structure F:\vlm datasets\Train\Train, let's just inspect it.
        # Actually there was F:\vlm datasets\Validation too. Let's check if that exists and has the same structure.
    }

    # 3. Load Data
    # Only loading Train for now to verify connectivity.
    # If Validation exists at F:\vlm datasets\Validation\Validation or similar, we could add it.

    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Dataset directory not found at {DATA_DIR}")
        return

    full_dataset = datasets.ImageFolder(DATA_DIR, data_transforms["train"])
    class_names = full_dataset.classes
    print(f"✅ Found {len(class_names)} classes: {class_names}")
    print(f"📊 Total Images: {len(full_dataset)}")

    # Split into Train/Val since we are pointing directly to the Train folder
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )  # 0 for Windows safety
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"📦 DataSplit: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # 4. Initialize Model (ResNet50)
    print("🧠 Initializing ResNet50...")
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 5. Training Loop
    since = time.time()
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                save_path = MODELS_DIR / "custom_vlm_best.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "classes": class_names,
                        "acc": best_acc,
                        "epoch": epoch,
                    },
                    save_path,
                )
                print(f"💾 Saved Best Model to {save_path}")

    time_elapsed = time.time() - since
    print(
        f"\n✅ Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    print(f"🎯 Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    train_custom_model()
