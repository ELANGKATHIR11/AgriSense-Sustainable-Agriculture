import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_image_folders(root_dir):
    """
    Recursively look for directories that look like they contain class folders (Healthy, Rust, Powdery)
    """
    candidates = []
    root = Path(root_dir)

    # Check if root itself has the classes
    if (
        (root / "Healthy").exists()
        or (root / "Rust").exists()
        or (root / "Powdery").exists()
    ):
        return root

    # Check subdirectories (depth 1)
    for path in root.iterdir():
        if path.is_dir():
            if (
                (path / "Healthy").exists()
                or (path / "Rust").exists()
                or (path / "Powdery").exists()
            ):
                return path

    # Check sub-subdirectories (depth 2 - handles vlm datasets/Train/Train)
    for path in root.iterdir():
        if path.is_dir():
            for subpath in path.iterdir():
                if subpath.is_dir():
                    if (subpath / "Healthy").exists() or (
                        subpath / "Rust"
                    ).exists():
                        return subpath
    return None


def train_model(data_dir, output_dir, num_epochs=10):
    print(f"Using device: {device}")

    # Data transformations
    data_transforms = {
        "Train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
        "Validation": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
        "Test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    # Locate dataset splits
    base_path = Path(data_dir)

    # Try to find standard structure
    train_dir = find_image_folders(base_path / "Train")
    val_dir = find_image_folders(base_path / "Validation")
    test_dir = find_image_folders(base_path / "Test")

    if not train_dir:
        print(f"Error: Could not find valid Train directory in {base_path}")
        return

    print(f"Training data found at: {train_dir}")
    if val_dir:
        print(f"Validation data found at: {val_dir}")

    image_datasets = {
        "Train": datasets.ImageFolder(train_dir, data_transforms["Train"])
    }
    if val_dir:
        image_datasets["Validation"] = datasets.ImageFolder(
            val_dir, data_transforms["Validation"]
        )

    dataloaders = {
        x: DataLoader(
            image_datasets[x], batch_size=32, shuffle=True, num_workers=0
        )
        for x in image_datasets
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    class_names = image_datasets["Train"].classes

    print(f"Classes found: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    # Save class names mapping
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "targeted_vlm_classes.json"), "w") as f:
        json.dump(class_names, f)

    # Load Pretrained ResNet
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Adjust final layer
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    since = time.time()

    best_model_wts = model_ft.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["Train", "Validation"]:
            if phase == "Validation" and "Validation" not in image_datasets:
                continue

            if phase == "Train":
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer_ft.zero_grad()

                with torch.set_grad_enabled(phase == "Train"):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "Train":
                        loss.backward()
                        optimizer_ft.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model
            if phase == "Validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model_ft.state_dict()

        print()

    time_elapsed = time.time() - since
    print(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    print(f"Best val Acc: {best_acc:4f}")

    # Load best model weights
    if "Validation" in image_datasets:
        model_ft.load_state_dict(best_model_wts)

    # Save model
    model_path = os.path.join(output_dir, "targeted_vlm_model.pth")
    torch.save(model_ft.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Targeted VLM (Rust/Powdery/Healthy)"
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))

    default_dataset = os.path.join(project_root, "vlm datasets")
    default_out = os.path.join(base_dir, "models")

    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Path to dataset root",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_out,
        help="Output directory for model",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs"
    )

    args = parser.parse_args()

    print("Starting Targeted VLM Training...")
    train_model(args.dataset, args.output, args.epochs)
