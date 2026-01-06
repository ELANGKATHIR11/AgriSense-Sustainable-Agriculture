"""
ML Dataset Preparation Script for AgriSense Crop Recommendation System
=======================================================================
This script processes raw crop data and generates ML-ready training datasets
for various agricultural prediction tasks:

1. Crop Recommendation - Given soil/climate conditions, recommend suitable crops
2. Crop Type Classification - Classify crops into categories (Cereal, Pulse, etc.)
3. Growth Duration Prediction - Predict how long a crop takes to mature
4. Water Requirement Estimation - Estimate daily water needs

Author: AgriSense Team
Date: 2026-01-05
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import pickle
from typing import Dict, Tuple, List, Any
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
ENCODERS_DIR = BASE_DIR / "encoders"

# Create directories
PROCESSED_DIR.mkdir(exist_ok=True)
ENCODERS_DIR.mkdir(exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    """Load the raw crop dataset."""
    csv_path = RAW_DATA_DIR / "india_crops_complete.csv"
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} crop records with {len(df.columns)} features")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for better ML performance.
    
    New Features:
    - temp_range: Temperature tolerance range
    - optimal_temp: Mid-point of temperature range
    - pH_range: Soil pH tolerance range
    - optimal_pH: Mid-point of pH range
    - rainfall_range: Rainfall tolerance range
    - avg_rainfall: Average rainfall requirement
    - moisture_range: Moisture tolerance range
    - avg_moisture: Average moisture requirement
    - npk_total: Total NPK requirement
    - npk_ratio_n: Nitrogen ratio in NPK
    - npk_ratio_p: Phosphorus ratio in NPK
    - npk_ratio_k: Potassium ratio in NPK
    - water_intensity: Water requirement category (low/medium/high)
    - duration_category: Growth duration category (short/medium/long)
    - is_perennial: Binary flag for perennial crops
    """
    df = df.copy()
    
    # Temperature features
    df['temp_range'] = df['max_temp_C'] - df['min_temp_C']
    df['optimal_temp'] = (df['min_temp_C'] + df['max_temp_C']) / 2
    
    # pH features
    df['pH_range'] = df['max_pH'] - df['min_pH']
    df['optimal_pH'] = (df['min_pH'] + df['max_pH']) / 2
    
    # Rainfall features
    df['rainfall_range'] = df['rainfall_max_mm'] - df['rainfall_min_mm']
    df['avg_rainfall'] = (df['rainfall_min_mm'] + df['rainfall_max_mm']) / 2
    
    # Moisture features
    df['moisture_range'] = df['moisture_max_percent'] - df['moisture_min_percent']
    df['avg_moisture'] = (df['moisture_min_percent'] + df['moisture_max_percent']) / 2
    
    # NPK features
    df['npk_total'] = df['N_kg_per_ha'] + df['P_kg_per_ha'] + df['K_kg_per_ha']
    df['npk_ratio_n'] = df['N_kg_per_ha'] / df['npk_total'].replace(0, 1)
    df['npk_ratio_p'] = df['P_kg_per_ha'] / df['npk_total'].replace(0, 1)
    df['npk_ratio_k'] = df['K_kg_per_ha'] / df['npk_total'].replace(0, 1)
    
    # Categorical derived features
    df['water_intensity'] = pd.cut(
        df['water_req_mm_day'],
        bins=[0, 4, 6, float('inf')],
        labels=['low', 'medium', 'high']
    )
    
    df['duration_category'] = pd.cut(
        df['growth_duration_days'],
        bins=[0, 90, 150, float('inf')],
        labels=['short', 'medium', 'long']
    )
    
    df['is_perennial'] = (df['season'] == 'Perennial').astype(int)
    
    print(f"✅ Engineered {12} new features")
    return df


def encode_categorical_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'label'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical features using specified method.
    
    Args:
        df: Input DataFrame
        columns: List of column names to encode
        method: 'label' for LabelEncoder, 'onehot' for OneHotEncoder
    
    Returns:
        Encoded DataFrame and dictionary of encoders
    """
    df = df.copy()
    encoders = {}
    
    if method == 'label':
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                print(f"  - {col}: {len(list(le.classes_))} classes")
    
    elif method == 'onehot':
        for col in columns:
            if col in df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df, dummies], axis=1)
                encoders[col] = list(dummies.columns)
                print(f"  - {col}: {len(dummies.columns)} one-hot columns")
    
    return df, encoders


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Safely split data with automatic stratification handling.
    Uses stratification if possible, falls back to random split.
    """
    try:
        # Check if stratification is possible
        unique, counts = np.unique(y, return_counts=True)
        min_count = min(counts)
        
        # Need at least 2 samples per class for stratification
        if min_count >= 2:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        else:
            print(f"    (Using random split: {sum(counts < 2)} classes have < 2 samples)")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
    except Exception as e:
        print(f"    (Falling back to random split: {str(e)[:50]}...)")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def create_crop_recommendation_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create dataset for crop recommendation task.
    
    Features: soil conditions, climate parameters, NPK levels
    Target: crop_name (multi-class classification)
    """
    print("\n📊 Creating Crop Recommendation Dataset...")
    
    # Feature columns for input (what farmer can measure/know)
    feature_cols = [
        'min_temp_C', 'max_temp_C', 'min_pH', 'max_pH',
        'rainfall_min_mm', 'rainfall_max_mm',
        'moisture_min_percent', 'moisture_max_percent',
        'N_kg_per_ha', 'P_kg_per_ha', 'K_kg_per_ha',
        'SOC_percent', 'optimal_temp', 'optimal_pH',
        'avg_rainfall', 'avg_moisture', 'npk_total',
        'soil_type_encoded', 'season_encoded'
    ]
    
    # Encode categorical columns first
    df_encoded, encoders = encode_categorical_features(
        df, ['crop_name', 'soil_type', 'season', 'crop_type'], method='label'
    )
    
    # Filter to available columns
    available_features = [c for c in feature_cols if c in df_encoded.columns]
    
    X = df_encoded[available_features].to_numpy()
    y = df_encoded['crop_name_encoded'].to_numpy()
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data (no stratification due to 96 unique crops - each appears only once)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    dataset = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': available_features,
        'target_name': 'crop_name',
        'scaler': scaler,
        'encoders': encoders,
        'n_classes': len(list(encoders['crop_name'].classes_)),
        'class_names': list(list(encoders['crop_name'].classes_))
    }
    
    print(f"  ✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  ✅ Features: {len(available_features)}, Classes: {dataset['n_classes']}")
    
    return dataset


def create_crop_type_classification_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create dataset for crop type classification.
    
    Features: all measurable parameters
    Target: crop_type (Cereal, Pulse, Vegetable, etc.)
    """
    print("\n📊 Creating Crop Type Classification Dataset...")
    
    feature_cols = [
        'min_temp_C', 'max_temp_C', 'min_pH', 'max_pH',
        'water_req_mm_day', 'rainfall_min_mm', 'rainfall_max_mm',
        'moisture_min_percent', 'moisture_max_percent',
        'N_kg_per_ha', 'P_kg_per_ha', 'K_kg_per_ha',
        'SOC_percent', 'growth_duration_days',
        'temp_range', 'optimal_temp', 'pH_range', 'optimal_pH',
        'rainfall_range', 'avg_rainfall', 'moisture_range', 'avg_moisture',
        'npk_total', 'npk_ratio_n', 'npk_ratio_p', 'npk_ratio_k'
    ]
    
    df_encoded, encoders = encode_categorical_features(
        df, ['crop_type', 'soil_type', 'season'], method='label'
    )
    
    available_features = [c for c in feature_cols if c in df_encoded.columns]
    
    X = df_encoded[available_features].to_numpy()
    y = df_encoded['crop_type_encoded'].to_numpy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = safe_train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    dataset = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': available_features,
        'target_name': 'crop_type',
        'scaler': scaler,
        'encoders': encoders,
        'n_classes': len(list(encoders['crop_type'].classes_)),
        'class_names': list(list(encoders['crop_type'].classes_))
    }
    
    print(f"  ✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  ✅ Features: {len(available_features)}, Classes: {dataset['n_classes']}")
    
    return dataset


def create_growth_duration_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create dataset for growth duration prediction (regression).
    
    Features: crop characteristics and environmental factors
    Target: growth_duration_days (continuous)
    """
    print("\n📊 Creating Growth Duration Prediction Dataset...")
    
    feature_cols = [
        'min_temp_C', 'max_temp_C', 'min_pH', 'max_pH',
        'water_req_mm_day', 'rainfall_min_mm', 'rainfall_max_mm',
        'moisture_min_percent', 'moisture_max_percent',
        'N_kg_per_ha', 'P_kg_per_ha', 'K_kg_per_ha',
        'SOC_percent', 'temp_range', 'optimal_temp',
        'pH_range', 'optimal_pH', 'avg_rainfall', 'avg_moisture',
        'npk_total', 'crop_type_encoded', 'season_encoded', 'soil_type_encoded'
    ]
    
    df_encoded, encoders = encode_categorical_features(
        df, ['crop_type', 'soil_type', 'season'], method='label'
    )
    
    available_features = [c for c in feature_cols if c in df_encoded.columns]
    
    X = df_encoded[available_features].to_numpy()
    y = df_encoded['growth_duration_days'].to_numpy()
    
    # Use MinMaxScaler for regression target
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    dataset = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_original': scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel(),
        'y_test_original': scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel(),
        'feature_names': available_features,
        'target_name': 'growth_duration_days',
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'encoders': encoders,
        'task_type': 'regression'
    }
    
    print(f"  ✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  ✅ Features: {len(available_features)}")
    print(f"  ✅ Target range: {y.min():.0f} - {y.max():.0f} days")
    
    return dataset


def create_water_requirement_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create dataset for water requirement estimation (regression).
    
    Features: environmental and soil conditions
    Target: water_req_mm_day (continuous)
    """
    print("\n📊 Creating Water Requirement Estimation Dataset...")
    
    feature_cols = [
        'min_temp_C', 'max_temp_C', 'min_pH', 'max_pH',
        'rainfall_min_mm', 'rainfall_max_mm',
        'moisture_min_percent', 'moisture_max_percent',
        'N_kg_per_ha', 'P_kg_per_ha', 'K_kg_per_ha',
        'SOC_percent', 'growth_duration_days',
        'temp_range', 'optimal_temp', 'avg_rainfall',
        'crop_type_encoded', 'season_encoded', 'soil_type_encoded'
    ]
    
    df_encoded, encoders = encode_categorical_features(
        df, ['crop_type', 'soil_type', 'season'], method='label'
    )
    
    available_features = [c for c in feature_cols if c in df_encoded.columns]
    
    X = df_encoded[available_features].to_numpy()
    y = df_encoded['water_req_mm_day'].to_numpy()
    
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    dataset = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_original': scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel(),
        'y_test_original': scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel(),
        'feature_names': available_features,
        'target_name': 'water_req_mm_day',
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'encoders': encoders,
        'task_type': 'regression'
    }
    
    print(f"  ✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  ✅ Features: {len(available_features)}")
    print(f"  ✅ Target range: {y.min():.1f} - {y.max():.1f} mm/day")
    
    return dataset


def create_season_classification_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create dataset for optimal season classification.
    
    Features: crop requirements
    Target: season (Kharif, Rabi, Zaid, Perennial, Kharif_Rabi)
    """
    print("\n📊 Creating Season Classification Dataset...")
    
    feature_cols = [
        'min_temp_C', 'max_temp_C', 'min_pH', 'max_pH',
        'water_req_mm_day', 'rainfall_min_mm', 'rainfall_max_mm',
        'moisture_min_percent', 'moisture_max_percent',
        'N_kg_per_ha', 'P_kg_per_ha', 'K_kg_per_ha',
        'SOC_percent', 'growth_duration_days',
        'temp_range', 'optimal_temp', 'pH_range',
        'avg_rainfall', 'npk_total', 'crop_type_encoded'
    ]
    
    df_encoded, encoders = encode_categorical_features(
        df, ['season', 'crop_type', 'soil_type'], method='label'
    )
    
    available_features = [c for c in feature_cols if c in df_encoded.columns]
    
    X = df_encoded[available_features].to_numpy()
    y = df_encoded['season_encoded'].to_numpy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = safe_train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    dataset = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': available_features,
        'target_name': 'season',
        'scaler': scaler,
        'encoders': encoders,
        'n_classes': len(list(encoders['season'].classes_)),
        'class_names': list(list(encoders['season'].classes_))
    }
    
    print(f"  ✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  ✅ Features: {len(available_features)}, Classes: {dataset['n_classes']}")
    print(f"  ✅ Seasons: {dataset['class_names']}")
    
    return dataset


def save_datasets(datasets: Dict[str, Dict], format: str = 'all'):
    """
    Save datasets in multiple formats for different ML frameworks.
    
    Formats:
    - pickle: For sklearn/PyTorch
    - csv: For general use and inspection
    - npz: NumPy compressed format
    """
    print("\n💾 Saving datasets...")
    
    for name, data in datasets.items():
        dataset_dir = PROCESSED_DIR / name
        dataset_dir.mkdir(exist_ok=True)
        
        # Save as pickle (complete with scalers and encoders)
        if format in ['all', 'pickle']:
            with open(dataset_dir / f'{name}_complete.pkl', 'wb') as f:
                pickle.dump(data, f)
            print(f"  ✅ Saved {name}_complete.pkl")
        
        # Save as CSV (for inspection)
        if format in ['all', 'csv']:
            # Training data
            train_df = pd.DataFrame(
                data['X_train'],
                columns=data['feature_names']
            )
            train_df['target'] = data['y_train']
            train_df.to_csv(dataset_dir / f'{name}_train.csv', index=False)
            
            # Test data
            test_df = pd.DataFrame(
                data['X_test'],
                columns=data['feature_names']
            )
            test_df['target'] = data['y_test']
            test_df.to_csv(dataset_dir / f'{name}_test.csv', index=False)
            print(f"  ✅ Saved {name}_train.csv and {name}_test.csv")
        
        # Save as NPZ (compressed NumPy)
        if format in ['all', 'npz']:
            np.savez_compressed(
                dataset_dir / f'{name}_data.npz',
                X_train=data['X_train'],
                X_test=data['X_test'],
                y_train=data['y_train'],
                y_test=data['y_test']
            )
            print(f"  ✅ Saved {name}_data.npz")
        
        # Save metadata as JSON
        metadata = {
            'feature_names': data['feature_names'],
            'target_name': data['target_name'],
            'n_train_samples': len(data['X_train']),
            'n_test_samples': len(data['X_test']),
            'n_features': len(data['feature_names']),
        }
        
        if 'n_classes' in data:
            metadata['n_classes'] = data['n_classes']
            metadata['class_names'] = data['class_names']
        
        if 'task_type' in data:
            metadata['task_type'] = data['task_type']
        
        with open(dataset_dir / f'{name}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Saved {name}_metadata.json")


def save_encoders(datasets: Dict[str, Dict]):
    """Save all encoders for inference use."""
    print("\n🔧 Saving encoders...")
    
    all_encoders = {}
    all_scalers = {}
    
    for name, data in datasets.items():
        if 'encoders' in data:
            for enc_name, encoder in data['encoders'].items():
                key = f"{name}_{enc_name}"
                if hasattr(encoder, 'classes_'):
                    all_encoders[key] = {
                        'type': 'LabelEncoder',
                        'classes': list(encoder.classes_)
                    }
        
        if 'scaler' in data:
            all_scalers[f"{name}_scaler"] = data['scaler']
        if 'scaler_X' in data:
            all_scalers[f"{name}_scaler_X"] = data['scaler_X']
        if 'scaler_y' in data:
            all_scalers[f"{name}_scaler_y"] = data['scaler_y']
    
    # Save encoder mappings as JSON
    with open(ENCODERS_DIR / 'label_encoders.json', 'w') as f:
        json.dump(all_encoders, f, indent=2)
    print("  ✅ Saved label_encoders.json")
    
    # Save scalers as pickle
    with open(ENCODERS_DIR / 'scalers.pkl', 'wb') as f:
        pickle.dump(all_scalers, f)
    print("  ✅ Saved scalers.pkl")


def generate_data_augmentation_samples(df: pd.DataFrame, n_augmented: int = 500) -> pd.DataFrame:
    """
    Generate augmented samples for training data expansion.
    
    Uses noise injection and interpolation to create synthetic samples.
    """
    print(f"\n🔀 Generating {n_augmented} augmented samples...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    augmented_rows = []
    
    np.random.seed(42)
    
    for _ in range(n_augmented):
        # Randomly select a base row
        base_idx = np.random.randint(0, len(df))
        base_row = df.iloc[base_idx].copy()
        
        # Add small noise to numeric columns
        for col in numeric_cols:
            noise_factor = 0.05  # 5% noise
            noise = base_row[col] * noise_factor * np.random.uniform(-1, 1)
            base_row[col] = base_row[col] + noise
        
        augmented_rows.append(base_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    print(f"  ✅ Generated {len(augmented_df)} augmented samples")
    
    return pd.concat([df, augmented_df], ignore_index=True)


def create_combined_ml_csv(df: pd.DataFrame):
    """
    Create a single comprehensive CSV file ready for ML training.
    All categorical variables encoded, all features engineered.
    """
    print("\n📄 Creating combined ML-ready CSV...")
    
    df_ml = df.copy()
    
    # Encode all categorical columns
    categorical_cols = ['crop_name', 'scientific_name', 'season', 'soil_type', 
                        'crop_type', 'water_intensity', 'duration_category']
    
    encoding_map = {}
    
    for col in categorical_cols:
        if col in df_ml.columns:
            le = LabelEncoder()
            df_ml[f'{col}_encoded'] = le.fit_transform(df_ml[col].astype(str))
            encoding_map[col] = dict(zip(list(le.classes_), list(le.transform(le.classes_))))
    
    # Save encoding mappings
    with open(PROCESSED_DIR / 'encoding_mappings.json', 'w') as f:
        json.dump(encoding_map, f, indent=2, default=str)
    
    # Save full ML-ready dataset
    df_ml.to_csv(PROCESSED_DIR / 'crops_ml_ready_full.csv', index=False)
    print(f"  ✅ Saved crops_ml_ready_full.csv ({len(df_ml)} rows, {len(df_ml.columns)} columns)")
    
    # Save numeric-only version (for quick model training)
    numeric_df = df_ml.select_dtypes(include=[np.number])
    numeric_df.to_csv(PROCESSED_DIR / 'crops_ml_numeric_only.csv', index=False)
    print(f"  ✅ Saved crops_ml_numeric_only.csv ({len(numeric_df.columns)} numeric columns)")
    
    return df_ml, encoding_map


def print_dataset_summary(datasets: Dict[str, Dict]):
    """Print a summary of all created datasets."""
    print("\n" + "="*70)
    print("📊 DATASET SUMMARY")
    print("="*70)
    
    for name, data in datasets.items():
        print(f"\n🔹 {name.upper()}")
        print(f"   Train samples: {len(data['X_train'])}")
        print(f"   Test samples: {len(data['X_test'])}")
        print(f"   Features: {len(data['feature_names'])}")
        
        if 'n_classes' in data:
            print(f"   Task: Classification ({data['n_classes']} classes)")
            print(f"   Classes: {data['class_names']}")
        else:
            print(f"   Task: Regression")
        
        print(f"   Feature columns: {data['feature_names'][:5]}...")


def main():
    """Main function to prepare all ML datasets."""
    print("="*70)
    print("🌾 AgriSense ML Dataset Preparation")
    print("="*70)
    
    # Step 1: Load raw data
    df = load_raw_data()
    
    # Step 2: Engineer features
    df = engineer_features(df)
    
    # Step 3: Create task-specific datasets
    datasets = {
        'crop_recommendation': create_crop_recommendation_dataset(df),
        'crop_type_classification': create_crop_type_classification_dataset(df),
        'growth_duration': create_growth_duration_dataset(df),
        'water_requirement': create_water_requirement_dataset(df),
        'season_classification': create_season_classification_dataset(df),
    }
    
    # Step 4: Save all datasets
    save_datasets(datasets)
    
    # Step 5: Save encoders
    save_encoders(datasets)
    
    # Step 6: Create combined ML CSV
    df_ml, _ = create_combined_ml_csv(df)
    
    # Step 7: Create augmented dataset
    df_augmented = generate_data_augmentation_samples(df)
    df_augmented = engineer_features(df_augmented)
    df_augmented.to_csv(PROCESSED_DIR / 'crops_augmented.csv', index=False)
    print(f"  ✅ Saved crops_augmented.csv ({len(df_augmented)} samples)")
    
    # Print summary
    print_dataset_summary(datasets)
    
    print("\n" + "="*70)
    print("✅ ML Dataset preparation complete!")
    print("="*70)
    print(f"\n📁 Output directories:")
    print(f"   Raw data: {RAW_DATA_DIR}")
    print(f"   Processed: {PROCESSED_DIR}")
    print(f"   Encoders: {ENCODERS_DIR}")
    
    return datasets, df_ml


if __name__ == "__main__":
    datasets, df_ml = main()
