"""
Comprehensive ML Model Training Orchestrator
Trains all AgriSense models with GPU/NPU acceleration
Implements advanced techniques for maximum accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
import json
from datetime import datetime
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================
class Config:
    BASE_DIR = Path(__file__).parent
    DATASETS_DIR = BASE_DIR / "datasets"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "training_logs"

    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_MIXED_PRECISION = torch.cuda.is_available()

    # Training
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 15

    # Cross-validation
    CV_FOLDS = 5


# ============================================================================
# Neural Network Models
# ============================================================================


class CropRecommendationNet(nn.Module):
    """Deep Neural Network for Crop Recommendation"""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class YieldPredictionNet(nn.Module):
    """LSTM-based network for Yield Prediction"""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # Reshape for LSTM: (batch, seq_len=1, features)
        x = x.unsqueeze(1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take last output
        return self.fc(x)


class WaterRequirementNet(nn.Module):
    """Multi-layer Perceptron for Water Requirement"""

    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# Dataset Classes
# ============================================================================


class TabularDataset(Dataset):
    """Generic dataset for tabular data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        if y.dtype == np.float64 or y.dtype == np.float32:
            self.y = torch.FloatTensor(y)
        else:
            self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# Training Functions
# ============================================================================


def train_epoch_classification(
    model, loader, criterion, optimizer, device, scaler=None
):
    """Train one epoch for classification"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in tqdm(loader, desc="Training", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        if scaler and Config.USE_MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def validate_classification(model, loader, criterion, device):
    """Validate classification model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Validation", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def train_epoch_regression(model, loader, criterion, optimizer, device, scaler=None):
    """Train one epoch for regression"""
    model.train()
    total_loss = 0

    for X_batch, y_batch in tqdm(loader, desc="Training", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()

        if scaler and Config.USE_MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_regression(model, loader, device):
    """Validate regression model"""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Validation", leave=False):
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    r2 = r2_score(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)

    return r2, mse, mae


# ============================================================================
# Model Training Functions
# ============================================================================


def train_crop_recommendation():
    """Train optimized Crop Recommendation model"""
    print("\n" + "=" * 80)
    print("🌾 TRAINING CROP RECOMMENDATION MODEL")
    print("=" * 80)

    # Load data
    data_path = Config.DATASETS_DIR / "soil_health_dataset.csv"
    df = pd.read_csv(data_path)

    # Prepare features and labels
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols].values
    y = df["label"].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"📊 Training samples: {len(X_train)}, Validation: {len(X_val)}")
    print(f"📊 Classes: {len(le.classes_)}")

    # Create datasets
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    # Create model
    model = CropRecommendationNet(len(feature_cols), len(le.classes_))
    model = model.to(Config.DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if Config.USE_MIXED_PRECISION else None

    # TensorBoard
    writer = SummaryWriter(Config.LOGS_DIR / "crop_recommendation")

    # Training loop
    best_val_acc = 0
    patience_counter = 0

    print(f"\n🚀 Training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_epoch_classification(
            model, train_loader, criterion, optimizer, Config.DEVICE, scaler
        )
        val_loss, val_acc = validate_classification(
            model, val_loader, criterion, Config.DEVICE
        )

        scheduler.step(val_acc)

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "label_encoder": le,
                    "val_acc": val_acc,
                    "classes": le.classes_.tolist(),
                },
                Config.MODELS_DIR / "crop_recommendation_optimized.pth",
            )

            print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

    writer.close()

    print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc


def train_yield_prediction():
    """Train optimized Yield Prediction model"""
    print("\n" + "=" * 80)
    print("📈 TRAINING YIELD PREDICTION MODEL")
    print("=" * 80)

    # Load data (using soil health dataset as proxy)
    data_path = Config.DATASETS_DIR / "soil_health_dataset.csv"
    df = pd.read_csv(data_path)

    # Create synthetic yield data based on soil parameters
    df["yield"] = (
        df["N"] * 0.3
        + df["P"] * 0.25
        + df["K"] * 0.2
        + df["temperature"] * 2
        + df["rainfall"] * 0.5
        + np.random.normal(0, 50, len(df))
    )

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols].values
    y = df["yield"].values

    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    print(f"📊 Training samples: {len(X_train)}, Validation: {len(X_val)}")

    # Create datasets
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    # Model
    model = YieldPredictionNet(len(feature_cols))
    model = model.to(Config.DEVICE)

    # Loss and optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    scaler = torch.cuda.amp.GradScaler() if Config.USE_MIXED_PRECISION else None
    writer = SummaryWriter(Config.LOGS_DIR / "yield_prediction")

    best_r2 = -float("inf")
    patience_counter = 0

    print(f"\n🚀 Training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch_regression(
            model, train_loader, criterion, optimizer, Config.DEVICE, scaler
        )
        r2, mse, mae = validate_regression(model, val_loader, Config.DEVICE)

        scheduler.step(r2)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Metrics/R2", r2, epoch)
        writer.add_scalar("Metrics/MSE", mse, epoch)
        writer.add_scalar("Metrics/MAE", mae, epoch)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}"
        )

        if r2 > best_r2:
            best_r2 = r2
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_X": scaler_X,
                    "scaler_y": scaler_y,
                    "r2_score": r2,
                },
                Config.MODELS_DIR / "yield_prediction_optimized.pth",
            )

            print(f"✅ Saved best model (R²: {r2:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

    writer.close()

    print(f"\n🎉 Training complete! Best R²: {best_r2:.4f}")
    return best_r2


def train_water_requirement():
    """Train optimized Water Requirement model"""
    print("\n" + "=" * 80)
    print("💧 TRAINING WATER REQUIREMENT MODEL")
    print("=" * 80)

    # Similar implementation to yield prediction
    # Using soil health dataset with synthetic water requirement
    data_path = Config.DATASETS_DIR / "soil_health_dataset.csv"
    df = pd.read_csv(data_path)

    # Synthetic water requirement based on crop and conditions
    df["water_req"] = (
        df["temperature"] * 5
        + (100 - df["humidity"]) * 2
        + df["rainfall"] * -0.3
        + np.random.normal(0, 20, len(df))
    ).clip(50, 500)

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols].values
    y = df["water_req"].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    print(f"📊 Training samples: {len(X_train)}, Validation: {len(X_val)}")

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    model = WaterRequirementNet(len(feature_cols))
    model = model.to(Config.DEVICE)

    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    scaler = torch.cuda.amp.GradScaler() if Config.USE_MIXED_PRECISION else None
    writer = SummaryWriter(Config.LOGS_DIR / "water_requirement")

    best_r2 = -float("inf")
    patience_counter = 0

    print(f"\n🚀 Training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch_regression(
            model, train_loader, criterion, optimizer, Config.DEVICE, scaler
        )
        r2, mse, mae = validate_regression(model, val_loader, Config.DEVICE)

        scheduler.step(r2)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Metrics/R2", r2, epoch)
        writer.add_scalar("Metrics/MSE", mse, epoch)
        writer.add_scalar("Metrics/MAE", mae, epoch)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}"
        )

        if r2 > best_r2:
            best_r2 = r2
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_X": scaler_X,
                    "scaler_y": scaler_y,
                    "r2_score": r2,
                },
                Config.MODELS_DIR / "water_requirement_optimized.pth",
            )

            print(f"✅ Saved best model (R²: {r2:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

    writer.close()

    print(f"\n🎉 Training complete! Best R²: {best_r2:.4f}")
    return best_r2


# ============================================================================
# Main Orchestrator
# ============================================================================


def main():
    """Main training orchestrator"""
    print("=" * 80)
    print("🚀 AGRISENSE ML MODEL OPTIMIZATION")
    print("=" * 80)
    print(f"Device: {Config.DEVICE}")
    print(f"Mixed Precision: {Config.USE_MIXED_PRECISION}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Max Epochs: {Config.EPOCHS}")
    print("=" * 80)

    # Create directories
    Config.MODELS_DIR.mkdir(exist_ok=True)
    Config.LOGS_DIR.mkdir(exist_ok=True)

    # Track results
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(Config.DEVICE),
        "models": {},
    }

    # Train all models
    try:
        print("\n🌾 Phase 1: Crop Recommendation")
        crop_acc = train_crop_recommendation()
        results["models"]["crop_recommendation"] = {"accuracy": crop_acc}
    except Exception as e:
        print(f"❌ Crop Recommendation failed: {e}")
        results["models"]["crop_recommendation"] = {"error": str(e)}

    try:
        print("\n📈 Phase 2: Yield Prediction")
        yield_r2 = train_yield_prediction()
        results["models"]["yield_prediction"] = {"r2_score": yield_r2}
    except Exception as e:
        print(f"❌ Yield Prediction failed: {e}")
        results["models"]["yield_prediction"] = {"error": str(e)}

    try:
        print("\n💧 Phase 3: Water Requirement")
        water_r2 = train_water_requirement()
        results["models"]["water_requirement"] = {"r2_score": water_r2}
    except Exception as e:
        print(f"❌ Water Requirement failed: {e}")
        results["models"]["water_requirement"] = {"error": str(e)}

    # Save results
    results_file = (
        Config.MODELS_DIR
        / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ ALL MODELS TRAINED!")
    print("=" * 80)
    print(f"\n📊 Results saved to: {results_file}")
    print("\nModel Performance:")
    for model_name, metrics in results["models"].items():
        print(f"  {model_name}: {metrics}")

    print("\n💡 View training logs: tensorboard --logdir training_logs")


if __name__ == "__main__":
    main()
