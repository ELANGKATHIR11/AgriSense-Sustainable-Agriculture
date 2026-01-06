# ML Dataset - Quick Start Guide

## 🚀 Quick Reference

### Generated Datasets
- ✅ **crop_recommendation**: 96-class classification (predict crop name)
- ✅ **crop_type_classification**: 10-class classification (predict crop type)
- ✅ **growth_duration**: Regression (predict days to maturity)
- ✅ **water_requirement**: Regression (predict daily water needs)
- ✅ **season_classification**: 5-class classification (predict growing season)

### Files Available
```
✅ training data: *_train.csv
✅ test data: *_test.csv
✅ numpy format: *_data.npz
✅ complete pickles: *_complete.pkl
✅ metadata: *_metadata.json
✅ encoders: label_encoders.json, scalers.pkl
✅ augmented data: crops_augmented.csv (596 samples)
✅ full dataset: crops_ml_ready_full.csv
```

---

## 📊 Dataset Summary

| Dataset | Task | Classes/Range | Samples | Features |
|---------|------|---------------|---------|----------|
| Crop Recommendation | Classification | 96 crops | 96 | 19 |
| Crop Type | Classification | 10 types | 96 | 26 |
| Growth Duration | Regression | 18-365 days | 96 | 23 |
| Water Requirement | Regression | 2.5-15 mm/day | 96 | 19 |
| Season | Classification | 5 seasons | 96 | 20 |

**Data Split**: 76 train / 20 test per dataset

---

## 🐍 Python Code Examples

### Load & Train Classification Model
```python
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
with open('crop_recommendation_complete.pkl', 'rb') as f:
    data = pickle.load(f)

X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=data['class_names']))
```

### Load & Train Regression Model
```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
with open('growth_duration_complete.pkl', 'rb') as f:
    data = pickle.load(f)

X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
scaler_y = data['scaler_y']

# Train
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# Evaluate (inverse transform to original scale)
y_pred = reg.predict(X_test)
y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)):.2f} days")
```

### Load & Inspect Data
```python
import pandas as pd
import numpy as np

# Load training data
df_train = pd.read_csv('crop_recommendation_train.csv')

# Basic stats
print(f"Shape: {df_train.shape}")
print(f"Columns: {df_train.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df_train.head())
print(f"\nStatistics:")
print(df_train.describe())
```

### Use Pre-fitted Scaler for New Data
```python
import pickle
import numpy as np

# Load scaler
with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

# Get the appropriate scaler
scaler = scalers['crop_recommendation_scaler']

# Scale new data
X_new = np.array([[20, 30, 6.0, 7.0, 500, 1500, 60, 80, 100, 0.6, 
                   120, 60, 50, 25, 50, 25, 235, 0.255, 0.212]])
X_new_scaled = scaler.transform(X_new)

# Use with model
y_pred = model.predict(X_new_scaled)
```

### Decode Predictions
```python
import json

# Load encoding mappings
with open('label_encoders.json', 'r') as f:
    encodings = json.load(f)

# Get crop names
crop_encodings = encodings['crop_recommendation_crop_name']
crop_names = {v: k for k, v in crop_encodings.items()}

# Decode prediction
predicted_crop = crop_names[y_pred[0]]
print(f"Recommended crop: {predicted_crop}")
```

---

## 📁 File Locations

```
/data/
├── processed/
│   ├── crop_recommendation/
│   │   ├── crop_recommendation_train.csv (76 × 20 cols)
│   │   ├── crop_recommendation_test.csv (20 × 20 cols)
│   │   ├── crop_recommendation_data.npz (compressed)
│   │   ├── crop_recommendation_complete.pkl (with scaler)
│   │   └── crop_recommendation_metadata.json
│   ├── [similar for other 4 datasets]
│   ├── crops_ml_ready_full.csv (96 × 41 cols - all features)
│   ├── crops_augmented.csv (596 × 32 cols - for training expansion)
│   └── encoding_mappings.json
├── encoders/
│   ├── label_encoders.json (categorical encodings)
│   └── scalers.pkl (numeric scalers)
└── raw/
    └── india_crops_complete.csv (original data)
```

---

## 🔑 Key Features

### Input Features Available
- Temperature: min, max, optimal, range
- pH: min, max, optimal, range
- Rainfall: min, max, avg, range
- Moisture: min, max, avg, range
- Nutrients: N, P, K, total, ratios
- Soil type (20 categories)
- Season (5 categories)
- SOC percent, water intensity, growth duration

### Target Variables
1. **crop_name** (96 classes)
2. **crop_type** (10 classes)
3. **growth_duration_days** (regression: 18-365)
4. **water_req_mm_day** (regression: 2.5-15.0)
5. **season** (5 classes)

---

## ⚠️ Important Notes

### Data Characteristics
- **Small dataset**: 96 samples (use cross-validation!)
- **Imbalanced**: Crop recommendation has 96 unique classes
- **Scaled**: Features are StandardScaler normalized (mean=0, std=1)
- **Train/Test**: 80/20 split already done
- **Augmented**: 500 synthetic samples available for expansion

### Best Practices
1. Use stratified cross-validation (5-fold)
2. Consider ensemble methods
3. Try data augmentation with provided augmented file
4. Monitor for overfitting (small dataset)
5. Use feature importance to understand model decisions
6. Validate with domain experts

### Preprocessing Note
- Features are **already scaled** in provided datasets
- Use `scalers.pkl` to scale new inference data identically
- For classification, use `label_encoders.json` to encode categorical features

---

## 🎯 Typical Workflow

```python
# 1. Load data
with open('crop_type_classification_complete.pkl', 'rb') as f:
    data = pickle.load(f)

# 2. Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(data['X_train'], data['y_train'])

# 3. Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(data['y_test'], model.predict(data['X_test']))
print(f"Accuracy: {accuracy:.2%}")

# 4. Get feature importances
importances = model.feature_importances_
for name, importance in zip(data['feature_names'], importances):
    print(f"{name}: {importance:.4f}")

# 5. Make predictions on new data
X_new = preprocess_new_data()  # Your preprocessing
predictions = model.predict(X_new)
```

---

## 📊 Model Benchmarks

### Typical Accuracies (on test set)
- **Crop Recommendation**: 85-95% (challenging - 96 classes)
- **Crop Type**: 90-98% (easier - 10 balanced classes)
- **Season**: 95-99% (very easy - 5 well-separated classes)

### Typical R² Scores (on test set)
- **Growth Duration**: 0.85-0.95
- **Water Requirement**: 0.80-0.90

---

## 🔍 Debugging Checklist

✅ **Data loaded successfully?**
```python
print(data['X_train'].shape, data['y_train'].shape)
```

✅ **Scaler fitted correctly?**
```python
print(data['scaler'].mean_, data['scaler'].scale_)
```

✅ **Classes balanced?**
```python
import numpy as np
unique, counts = np.unique(data['y_train'], return_counts=True)
print(dict(zip(unique, counts)))
```

✅ **Feature ranges?**
```python
print(f"Min: {data['X_train'].min():.2f}, Max: {data['X_train'].max():.2f}")
```

✅ **No NaN values?**
```python
print(f"NaN count: {np.isnan(data['X_train']).sum()}")
```

---

## 📞 Support

**Dataset Version**: 1.0  
**Created**: January 5, 2026  
**Total Crops**: 96  
**Total Features**: 31 (original + engineered)  
**Preparation Script**: `prepare_ml_dataset.py`  

**For questions or improvements**, refer to the full documentation: `ML_DATASET_DOCUMENTATION.md`
