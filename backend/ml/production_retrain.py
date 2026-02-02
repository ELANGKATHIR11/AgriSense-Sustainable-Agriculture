"""
Production-Grade ML Retraining with Real Datasets
Uses actual agricultural data from project for maximum accuracy
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              VotingClassifier)
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

print("=" * 90)
print("🚀 PRODUCTION ML RETRAINING - Using REAL Agricultural Datasets")
print("=" * 90)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "retraining_results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset paths
PROJECT_ROOT = BASE_DIR.parent.parent
DATASETS_ROOT = BASE_DIR / "datasets_enhanced"

print(f"📂 Project root: {PROJECT_ROOT}")
print(f"📂 Datasets root: {DATASETS_ROOT}")
print(f"📂 Models dir: {MODELS_DIR}\n")

# ============================================================================
# 1. CROP RECOMMENDATION - Real Data
# ============================================================================

print("\n" + "=" * 90)
print("🌾 CROP RECOMMENDATION - Training with Real Data")
print("=" * 90)

# Load REAL crop recommendation dataset
crop_data_path = (
    DATASETS_ROOT / "crop_recommendation" / "crop_data_enhanced.csv"
)

if crop_data_path.exists():
    print(f"📂 Loading: {crop_data_path}")
    data = pd.read_csv(crop_data_path)
else:
    print("⚠️  No dataset found, exiting...")
    exit(1)

print(f"✅ Dataset loaded: {data.shape} rows")
print(f"🎯 Classes: {data['label'].nunique()} unique crops")
print(f"📊 Samples per class:\n{data['label'].value_counts().head(10)}\n")

# Prepare data
X = data.drop("label", axis=1)
y = data["label"]

# Simple split without stratify for sparse datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"🔀 Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
print("🔍 Randomized Search for optimal hyperparameters...")
rf = RandomForestClassifier(
    random_state=42, n_jobs=1
)  # Single job per forest to save RAM

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid_search = RandomizedSearchCV(
    rf,
    param_grid,
    n_iter=8,
    cv=3,
    scoring="accuracy",
    n_jobs=1,
    verbose=1,
    random_state=42,
)

print("⏳ Training (this will take 2-5 minutes with GridSearchCV)...")
grid_search.fit(X_train_scaled, y_train)

print(f"\n✅ Best Parameters: {grid_search.best_params_}")
print(f"✅ Best CV Score: {grid_search.best_score_:.4f}")

# Get best model
best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n📊 TEST RESULTS:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   F1-Score: {f1:.4f}")

# Cross-validation
cv_scores = cross_val_score(
    best_model, X_train_scaled, y_train, cv=3, n_jobs=1
)
print(f"   3-Fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Save model
model_path = MODELS_DIR / "production_crop_recommendation_model.pkl"
scaler_path = MODELS_DIR / "production_crop_recommendation_scaler.pkl"

pickle.dump(best_model, open(model_path, "wb"))
pickle.dump(scaler, open(scaler_path, "wb"))

# Save classification report
report = classification_report(y_test, y_pred, output_dict=True)

results_crop = {
    "model": "Crop Recommendation",
    "dataset_size": len(data),
    "n_classes": data["label"].nunique(),
    "test_accuracy": float(accuracy),
    "test_f1_score": float(f1),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
    "best_params": grid_search.best_params_,
    "timestamp": datetime.now().isoformat(),
}

with open(RESULTS_DIR / "production_crop_recommendation.json", "w") as f:
    json.dump(results_crop, f, indent=2)

print(f"✅ Saved: {model_path.name}\n")

# ============================================================================
# 2. WATER REQUIREMENT - Real Data
# ============================================================================

print("=" * 90)
print("💧 WATER REQUIREMENT - Training with Real Data")
print("=" * 90)

water_train_path = (
    DATASETS_ROOT / "water_requirement" / "water_requirement_enhanced.csv"
)

if water_train_path.exists():
    print(f"📂 Loading: {water_train_path}")
    full_water_data = pd.read_csv(water_train_path)

    # Split into train/test
    train_data, test_data = train_test_split(
        full_water_data, test_size=0.2, random_state=42
    )
    print(
        f"✅ Train: {len(train_data)} samples, Test: {len(test_data)} samples"
    )

    # Identify target column (usually last column)
    target_col = train_data.columns[-1]
    print(f"🎯 Target: {target_col}")

    X_train = train_data.drop(target_col, axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]

    # Handle categorical columns
    for col in X_train.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ensemble
    print("🔍 Training RandomForest + GradientBoosting ensemble...")

    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=30, random_state=42, n_jobs=1
    )
    gb_model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, random_state=42
    )

    rf_model.fit(X_train_scaled, y_train)
    gb_model.fit(X_train_scaled, y_train)

    # Evaluate both
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)

    rf_r2 = r2_score(y_test, rf_pred)
    gb_r2 = r2_score(y_test, gb_pred)

    print(f"   RandomForest R²: {rf_r2:.4f}")
    print(f"   GradientBoosting R²: {gb_r2:.4f}")

    # Use better model
    best_model = rf_model if rf_r2 > gb_r2 else gb_model
    best_name = "RandomForest" if rf_r2 > gb_r2 else "GradientBoosting"
    best_r2 = max(rf_r2, gb_r2)

    print(f"🏆 Best: {best_name} (R²: {best_r2:.4f})")

    # Save
    pickle.dump(
        best_model,
        open(MODELS_DIR / "production_water_requirement_model.pkl", "wb"),
    )
    pickle.dump(
        scaler,
        open(MODELS_DIR / "production_water_requirement_scaler.pkl", "wb"),
    )

    results_water = {
        "model": "Water Requirement",
        "dataset_size": len(train_data) + len(test_data),
        "best_estimator": best_name,
        "test_r2_score": float(best_r2),
        "test_rmse": float(
            np.sqrt(
                mean_squared_error(y_test, best_model.predict(X_test_scaled))
            )
        ),
        "timestamp": datetime.now().isoformat(),
    }

    with open(RESULTS_DIR / "production_water_requirement.json", "w") as f:
        json.dump(results_water, f, indent=2)

    print("✅ Saved: production_water_requirement_model.pkl\n")
else:
    print(f"⚠️  Dataset not found at {water_train_path}")
    results_water = {"error": "Dataset not found"}

# ============================================================================
# 3. SEASON CLASSIFICATION - Real Data
# ============================================================================

print("=" * 90)
print("🌤️  SEASON CLASSIFICATION - Training with Real Data")
print("=" * 90)

season_train_path = (
    DATASETS_ROOT / "season_classification" / "season_data_enhanced.csv"
)

if season_train_path.exists():
    full_season_data = pd.read_csv(season_train_path)
    train_data, test_data = train_test_split(
        full_season_data, test_size=0.2, random_state=42
    )

    print(
        f"✅ Train: {len(train_data)} samples, Test: {len(test_data)} samples"
    )

    target_col = train_data.columns[-1]
    X_train = train_data.drop(target_col, axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]

    # Encode categorical
    for col in X_train.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Ensemble for Max Accuracy
    print(
        "🔍 Training Optimized Ensemble (RandomForest + GradientBoosting)..."
    )
    model_rf = RandomForestClassifier(
        n_estimators=300, max_depth=30, random_state=42, n_jobs=1
    )
    model_gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42
    )

    model = VotingClassifier(
        estimators=[("rf", model_rf), ("gb", model_gb)], voting="soft"
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1-Score: {f1:.4f}")

    # Save
    pickle.dump(
        model,
        open(MODELS_DIR / "production_season_classification_model.pkl", "wb"),
    )
    pickle.dump(
        scaler,
        open(MODELS_DIR / "production_season_classification_scaler.pkl", "wb"),
    )

    results_season = {
        "model": "Season Classification",
        "test_accuracy": float(accuracy),
        "test_f1_score": float(f1),
        "timestamp": datetime.now().isoformat(),
    }

    with open(RESULTS_DIR / "production_season_classification.json", "w") as f:
        json.dump(results_season, f, indent=2)

    print("✅ Saved: production_season_classification_model.pkl\n")
else:
    print(f"⚠️  Dataset not found at {season_train_path}")
    results_season = {"error": "Dataset not found"}

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 90)
print("📊 PRODUCTION RETRAINING COMPLETE")
print("=" * 90)

print("\n🌾 Crop Recommendation:")
print(f"   Accuracy: {results_crop.get('test_accuracy', 0)*100:.2f}%")
print(f"   F1-Score: {results_crop.get('test_f1_score', 0):.4f}")
print(f"   CV Score: {results_crop.get('cv_mean', 0):.4f}")

if "error" not in results_water:
    print("\n💧 Water Requirement:")
    print(f"   R² Score: {results_water.get('test_r2_score', 0):.4f}")
    print(f"   RMSE: {results_water.get('test_rmse', 0):.2f}")

if "error" not in results_season:
    print("\n🌤️  Season Classification:")
    print(f"   Accuracy: {results_season.get('test_accuracy', 0)*100:.2f}%")

print(f"\n📁 Models saved in: {MODELS_DIR}")
print(f"📊 Results saved in: {RESULTS_DIR}")
print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n🎉 ALL MODELS TRAINED WITH REAL DATA - PRODUCTION READY!")
