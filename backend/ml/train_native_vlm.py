"""
Native Vision-Language Model (VLM) for Plant Disease Detection
Using PyTorch with CLIP-style architecture

Based on latest research (2024-2025):
- Lightweight CLIP architecture for agricultural applications
- Multi-modal learning (image + text) for disease classification
- Optimized for CPU/edge deployment
- Training on PlantVillage and custom agricultural datasets
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor


# Configuration
class VLMConfig:
    # Model architecture
    image_encoder = "resnet50"  # Lightweight backbone
    text_encoder = "distilbert-base-uncased"
    embedding_dim = 512

    # Training
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 1e-5

    # Dataset
    img_size = 224
    num_classes = 38  # PlantVillage has 38 disease classes

    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "datasets" / "plant_disease_vlm"
    model_dir = base_dir / "models"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


config = VLMConfig()

# ============================================================================
# 1. IMAGE ENCODER (Vision)
# ============================================================================


class ImageEncoder(nn.Module):
    """Lightweight image encoder based on ResNet50"""

    def __init__(self, embedding_dim=512):
        super().__init__()
        # Use pretrained ResNet50
        resnet = models.resnet50(pretrained=True)

        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Projection head to embedding space
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, images):
        features = self.features(images)
        embeddings = self.projection(features)
        # L2 normalize
        return F.normalize(embeddings, dim=-1)


# ============================================================================
# 2. TEXT ENCODER (Language)
# ============================================================================


class TextEncoder(nn.Module):
    """Simple text encoder for disease descriptions"""

    def __init__(self, vocab_size=10000, embedding_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, text_tokens):
        # text_tokens: [batch, seq_len]
        embedded = self.embedding(text_tokens)
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)

        embeddings = self.projection(hidden)
        # L2 normalize
        return F.normalize(embeddings, dim=-1)


# ============================================================================
# 3. VLM MODEL (Combined)
# ============================================================================


class PlantDiseaseVLM(nn.Module):
    """Vision-Language Model for Plant Disease Detection"""

    def __init__(self, num_classes=38, embedding_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        self.text_encoder = TextEncoder(embedding_dim=embedding_dim)

        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, images, text_tokens=None, return_embeddings=False):
        # Encode images
        image_embeddings = self.image_encoder(images)

        if return_embeddings:
            return image_embeddings

        # Classification
        logits = self.classifier(image_embeddings)

        # If text provided, also compute contrastive loss
        if text_tokens is not None:
            text_embeddings = self.text_encoder(text_tokens)

            # Compute similarity matrix
            logits_per_image = (
                image_embeddings @ text_embeddings.t() * self.temperature.exp()
            )
            logits_per_text = logits_per_image.t()

            return logits, logits_per_image, logits_per_text

        return logits


# ============================================================================
# 4. DATASET
# ============================================================================


class PlantDiseaseDataset(Dataset):
    """Dataset for plant disease images with text descriptions"""

    def __init__(self, root_dir, metadata_file, transform=None, is_train=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train

        # Load metadata
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        self.samples = self.metadata["samples"]
        self.class_names = self.metadata["class_names"]
        self.disease_descriptions = self.metadata.get("descriptions", {})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img_path = self.root_dir / sample["image_path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get label
        label = sample["class_id"]

        # Get text description
        class_name = self.class_names[label]
        description = self.disease_descriptions.get(
            class_name, f"A plant leaf showing {class_name}"
        )

        return {
            "image": image,
            "label": label,
            "text": description,
            "class_name": class_name,
        }


# ============================================================================
# 5. TRAINING
# ============================================================================


def create_dataloaders(config):
    """Create train and validation dataloaders"""

    # Image transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = PlantDiseaseDataset(
        root_dir=config.data_dir / "images",
        metadata_file=config.data_dir / "train_metadata.json",
        transform=train_transform,
        is_train=True,
    )

    val_dataset = PlantDiseaseDataset(
        root_dir=config.data_dir / "images",
        metadata_file=config.data_dir / "val_metadata.json",
        transform=val_transform,
        is_train=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            {"loss": total_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
        )

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(val_loader), 100.0 * correct / total


def train_vlm(config):
    """Main training function"""
    print("=" * 80)
    print("🌱 Training Native Plant Disease VLM")
    print("=" * 80)

    # Create model
    model = PlantDiseaseVLM(
        num_classes=config.num_classes, embedding_dim=config.embedding_dim
    ).to(config.device)

    print(
        f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    print(f"💻 Device: {config.device}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    print(f"📊 Train samples: {len(train_loader.dataset)}")
    print(f"📊 Val samples: {len(val_loader.dataset)}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )

    # Training loop
    best_val_acc = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, config.device, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, config.device
        )

        # Update scheduler
        scheduler.step()

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "config": config.__dict__,
                },
                config.model_dir / "best_vlm_model.pth",
            )
            print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")

    # Save final model
    torch.save(model.state_dict(), config.model_dir / "final_vlm_model.pth")

    # Save training history
    with open(config.model_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print(f"🎉 Training Complete! Best Val Acc: {best_val_acc:.2f}%")
    print("=" * 80)

    return model, history


# ============================================================================
# 6. INFERENCE
# ============================================================================


class VLMInference:
    """Inference wrapper for the trained VLM"""

    def __init__(self, model_path, config):
        self.config = config
        self.device = config.device

        # Load model
        self.model = PlantDiseaseVLM(
            num_classes=config.num_classes, embedding_dim=config.embedding_dim
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        # Load class names
        metadata_path = config.data_dir / "train_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.class_names = metadata["class_names"]
        self.disease_info = metadata.get("disease_info", {})

        # Image transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((config.img_size, config.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image_path, top_k=3):
        """Predict disease from image"""
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(top_k, dim=-1)

        # Format results
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = self.class_names[idx.item()]
            results.append(
                {
                    "disease": class_name,
                    "confidence": prob.item(),
                    "treatment": self.disease_info.get(class_name, {}).get(
                        "treatment", "N/A"
                    ),
                    "severity": self._assess_severity(prob.item()),
                }
            )

        return results

    def _assess_severity(self, confidence):
        if confidence > 0.8:
            return "High confidence detection"
        elif confidence > 0.5:
            return "Moderate confidence"
        else:
            return "Low confidence - verify manually"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Plant Disease VLM")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--image", type=str, help="Image path for inference")
    parser.add_argument("--model", type=str, help="Model path for inference")
    args = parser.parse_args()

    if args.mode == "train":
        model, history = train_vlm(config)
        print("\n✅ Model training complete!")
        print(f"📁 Saved to: {config.model_dir}")

    elif args.mode == "infer":
        if not args.image or not args.model:
            print("❌ Please provide --image and --model for inference")
            exit(1)

        predictor = VLMInference(args.model, config)
        results = predictor.predict(args.image)

        print("\n🔍 Prediction Results:")
        print("=" * 60)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['disease']}")
            print(f"   Confidence: {result['confidence']*100:.2f}%")
            print(f"   Severity: {result['severity']}")
            print(f"   Treatment: {result['treatment']}")
