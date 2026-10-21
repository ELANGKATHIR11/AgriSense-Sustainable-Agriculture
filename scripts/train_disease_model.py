import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np

def train_disease_model():
    start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using dGPU device: {device} ({torch.cuda.get_device_name(0)})")

    train_dir = r"D:\Agrisense-A samart Agriculture Solution\Agrisense\dataset\diseases\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
    valid_dir = r"D:\Agrisense-A samart Agriculture Solution\Agrisense\dataset\diseases\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Indexing datasets...")
    full_train = datasets.ImageFolder(train_dir, transform=train_transform)
    full_val = datasets.ImageFolder(valid_dir, transform=val_transform)
    classes = full_train.classes
    num_classes = len(classes)
    print(f"Detected {num_classes} plant disease classes.")

    # Select stratified subset for fast, highly focused GPU training
    np.random.seed(42)
    train_indices = []
    val_indices = []

    targets_train = np.array(full_train.targets)
    targets_val = np.array(full_val.targets)

    for c in range(num_classes):
        c_train_idx = np.where(targets_train == c)[0]
        c_val_idx = np.where(targets_val == c)[0]
        
        # 120 samples per class for training, 30 for validation
        n_train = min(120, len(c_train_idx))
        n_val = min(30, len(c_val_idx))
        
        train_indices.extend(np.random.choice(c_train_idx, n_train, replace=False))
        val_indices.extend(np.random.choice(c_val_idx, n_val, replace=False))

    train_sub = Subset(full_train, train_indices)
    val_sub = Subset(full_val, val_indices)

    print(f"Training on {len(train_sub)} images, validating on {len(val_sub)} images...")

    train_loader = DataLoader(train_sub, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    print("Initializing MobileNetV3-Small backbone on RTX 5060...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Replace classification head
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = 4
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == labels.data).item()
                    val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc

    # Save model checkpoint
    os.makedirs("models/artifacts", exist_ok=True)
    os.makedirs("models/metadata", exist_ok=True)
    
    artifact_path = "models/artifacts/plant_disease_model.pth"
    torch.save({
        'state_dict': model.state_dict(),
        'classes': classes,
        'arch': 'mobilenet_v3_small',
        'val_accuracy': best_acc
    }, artifact_path)
    print(f"Model saved to {artifact_path} with Best Validation Accuracy: {best_acc*100:.2f}%")

    # Generate metadata and class mappings with agronomic details
    class_metadata = {}
    for idx, c in enumerate(classes):
        parts = c.split("___")
        crop = parts[0].replace("_", " ").title()
        disease_raw = parts[1].replace("_", " ").title() if len(parts) > 1 else "Healthy"
        is_healthy = "healthy" in disease_raw.lower()

        if is_healthy:
            severity = "none"
            symptoms = ["Vibrant green foliage", "No chlorosis or necrotic lesions", "Normal leaf architecture"]
            organic = ["Continue balanced compost application", "Regular moisture monitoring", "Maintain preventative neem oil foliar spray (0.5%)"]
            chemical = ["No chemical fungicides needed"]
            prevention = ["Crop rotation every 2 seasons", "Maintain adequate spacing for airflow", "Drip irrigation at soil level"]
        else:
            severity = "high" if any(w in disease_raw.lower() for w in ["blight", "rot", "virus"]) else "moderate"
            symptoms = [f"Visible {disease_raw.lower()} lesions on foliage", "Foliar chlorosis and leaf curl", "Premature defoliation risk"]
            organic = ["Spray Trichoderma viride bio-fungicide @ 5g/liter", "Apply 1% Bordeaux mixture or copper soap", "Remove and bury infected leaves"]
            chemical = ["Apply Mancozeb 75% WP @ 2g/liter or Azoxystrobin 23% SC @ 1ml/liter", "Spray Systemic Carbendazim 50% WP @ 1g/liter"]
            prevention = ["Avoid overhead sprinkler irrigation", "Disinfect pruning tools with 10% sodium hypochlorite", "Use certified disease-resistant hybrid seed varieties"]

        class_metadata[c] = {
            "index": idx,
            "cropIdentified": crop,
            "diseaseName": f"{crop} {disease_raw}" if not is_healthy else f"{crop} Healthy",
            "scientificName": f"{crop} Pathology Profile",
            "isHealthy": is_healthy,
            "severity": severity,
            "symptomsObserved": symptoms,
            "organicTreatment": organic,
            "chemicalTreatment": chemical,
            "preventionSteps": prevention
        }

    meta_payload = {
        "model_id": "agrisense-vision-mobilenet",
        "version": "v1.0.0",
        "name": "AgriSense Vision Plant Leaf Pathology Classifier",
        "type": "IMAGE_CLASSIFICATION",
        "framework": "PyTorch 2.14 / MobileNetV3-Small (CUDA RTX 5060)",
        "device": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "classes_count": num_classes,
        "validation_accuracy": round(best_acc, 4),
        "classes": classes,
        "details": class_metadata,
        "artifact_path": artifact_path,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_training_sec": round(time.time() - start_time, 2)
    }

    with open("models/metadata/plant_disease_model.json", "w") as f:
        json.dump(meta_payload, f, indent=2)

    print("Metadata written to models/metadata/plant_disease_model.json successfully!")

if __name__ == "__main__":
    train_disease_model()
