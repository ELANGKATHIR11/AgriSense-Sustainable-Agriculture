# ML Dataset - Complete File Index

**Generated**: January 5, 2026  
**Total Files**: 36  
**Total Size**: ~2-3 MB (compressed datasets)  

---

## 📋 File Inventory

### Core Documentation (3 files)
```
✅ GENERATION_SUMMARY.md              Overview of what was created
✅ ML_DATASET_DOCUMENTATION.md        Complete 50+ page documentation
✅ QUICK_START_GUIDE.md               Quick reference & code examples
```

### Raw Data (1 file)
```
📁 raw/
   └── india_crops_complete.csv       96 crops, 19 features (~15 KB)
```

### Processed Data (27 files)

#### Combined Datasets (3 files)
```
📁 processed/
   ├── crops_ml_ready_full.csv        All features, all crops (96 × 41 cols)
   ├── crops_ml_numeric_only.csv      Numeric features only (34 cols)
   └── crops_augmented.csv            596 samples (96 + 500 synthetic)
```

#### Crop Recommendation (5 files)
```
   ├── crop_recommendation/
   │   ├── crop_recommendation_train.csv         76 × 20 columns
   │   ├── crop_recommendation_test.csv          20 × 20 columns
   │   ├── crop_recommendation_data.npz          NumPy format
   │   ├── crop_recommendation_complete.pkl      With scaler
   │   └── crop_recommendation_metadata.json     Feature info
```

#### Crop Type Classification (5 files)
```
   ├── crop_type_classification/
   │   ├── crop_type_classification_train.csv    76 × 27 columns
   │   ├── crop_type_classification_test.csv     20 × 27 columns
   │   ├── crop_type_classification_data.npz     NumPy format
   │   ├── crop_type_classification_complete.pkl With scaler
   │   └── crop_type_classification_metadata.json
```

#### Growth Duration Prediction (5 files)
```
   ├── growth_duration/
   │   ├── growth_duration_train.csv             76 × 24 columns
   │   ├── growth_duration_test.csv              20 × 24 columns
   │   ├── growth_duration_data.npz              NumPy format
   │   ├── growth_duration_complete.pkl          With scalers
   │   └── growth_duration_metadata.json
```

#### Water Requirement Prediction (5 files)
```
   ├── water_requirement/
   │   ├── water_requirement_train.csv           76 × 20 columns
   │   ├── water_requirement_test.csv            20 × 20 columns
   │   ├── water_requirement_data.npz            NumPy format
   │   ├── water_requirement_complete.pkl        With scalers
   │   └── water_requirement_metadata.json
```

#### Season Classification (5 files)
```
   ├── season_classification/
   │   ├── season_classification_train.csv       76 × 21 columns
   │   ├── season_classification_test.csv        20 × 21 columns
   │   ├── season_classification_data.npz        NumPy format
   │   ├── season_classification_complete.pkl    With scaler
   │   └── season_classification_metadata.json
   │
   └── encoding_mappings.json                    Categorical encodings
```

### Encoders & Scalers (2 files)
```
📁 encoders/
   ├── label_encoders.json             Mappings: crop_name, crop_type, season, soil_type
   └── scalers.pkl                     Pre-fitted: StandardScaler, MinMaxScaler
```

### Utility Script (1 file)
```
prepare_ml_dataset.py                  Reproducible generation script (~700 lines)
```

---

## 📊 Data Format Summary

### CSV Files (23 files)
**Format**: Comma-separated values, readable text  
**Use**: Excel, Pandas, general analysis  
**Samples**:
- Train files: 76 rows each
- Test files: 20 rows each
- Full files: 96 rows
- Augmented: 596 rows

**Size**: ~5-20 KB each

**Example Train File Structure**:
```
feature_1,feature_2,...,feature_N,target
value_1,value_2,...,value_N,target_value
...
```

### NPZ Files (5 files)
**Format**: NumPy compressed binary  
**Use**: NumPy operations, efficient storage  
**Contents**: X_train, X_test, y_train, y_test arrays

**Size**: ~5-10 KB each (highly compressed)

**Load Example**:
```python
data = np.load('crop_recommendation_data.npz')
X_train = data['X_train']  # Access arrays
```

### Pickle Files (6 files)
**Format**: Python object serialization  
**Use**: scikit-learn, production deployment  
**Contents**: Complete datasets with scalers & encoders

**Size**: ~50-100 KB each

**Load Example**:
```python
with open('crop_recommendation_complete.pkl', 'rb') as f:
    data = pickle.load(f)
```

### JSON Files (7 files)
**Format**: JavaScript Object Notation  
**Use**: Metadata, configuration, inspection  
**Contents**: Feature names, class names, encoding mappings

**Size**: ~2-5 KB each

---

## 🎯 Which Files to Use?

### For Quick Start
- **Start Here**: `QUICK_START_GUIDE.md`
- **Load Data**: `*_complete.pkl` files
- **Code Examples**: Copy from guide

### For Data Exploration
- **Inspect Data**: `*_train.csv` files (Excel/Pandas)
- **Check Stats**: `*_metadata.json` files
- **Full Data**: `crops_ml_ready_full.csv`

### For ML Models
- **Training**: Use `*_train.csv` or `*_complete.pkl`
- **Testing**: Use `*_test.csv` or `*_complete.pkl`
- **Scaling**: Pre-loaded in `*_complete.pkl`
- **Encoding**: In `encoders/label_encoders.json`

### For Production Deployment
- **Load Scaler**: `encoders/scalers.pkl`
- **Load Model**: Saved `.pkl` or `.joblib`
- **Encode Input**: `encoders/label_encoders.json`
- **Make Predictions**: Pipeline with loaded scaler

### For Efficiency
- **Fastest Load**: Use `*.npz` files
- **Smallest Files**: Use compressed `.npz`
- **Most Flexible**: Use `*_complete.pkl`
- **Most Readable**: Use `*.csv`

---

## 📈 Dataset Specifications

### Crop Recommendation
| Property | Value |
|----------|-------|
| File Pattern | `crop_recommendation_*` |
| Task | 96-class classification |
| Train Samples | 76 |
| Test Samples | 20 |
| Features | 19 |
| Target | crop_name (96 classes) |
| CSV File Size | ~12 KB (train), ~3 KB (test) |
| NPZ File Size | ~8 KB |

### Crop Type Classification
| Property | Value |
|----------|-------|
| File Pattern | `crop_type_classification_*` |
| Task | 10-class classification |
| Train Samples | 76 |
| Test Samples | 20 |
| Features | 26 |
| Target | crop_type (10 classes) |
| CSV File Size | ~15 KB (train), ~4 KB (test) |
| NPZ File Size | ~10 KB |

### Growth Duration
| Property | Value |
|----------|-------|
| File Pattern | `growth_duration_*` |
| Task | Regression |
| Train Samples | 76 |
| Test Samples | 20 |
| Features | 23 |
| Target | growth_duration_days (18-365) |
| CSV File Size | ~13 KB (train), ~3 KB (test) |
| NPZ File Size | ~9 KB |

### Water Requirement
| Property | Value |
|----------|-------|
| File Pattern | `water_requirement_*` |
| Task | Regression |
| Train Samples | 76 |
| Test Samples | 20 |
| Features | 19 |
| Target | water_req_mm_day (2.5-15.0) |
| CSV File Size | ~10 KB (train), ~2.5 KB (test) |
| NPZ File Size | ~7 KB |

### Season Classification
| Property | Value |
|----------|-------|
| File Pattern | `season_classification_*` |
| Task | 5-class classification |
| Train Samples | 76 |
| Test Samples | 20 |
| Features | 20 |
| Target | season (5 classes) |
| CSV File Size | ~11 KB (train), ~3 KB (test) |
| NPZ File Size | ~8 KB |

---

## 🔑 Key Files Explained

### QUICK_START_GUIDE.md
- 📖 Quick reference guide
- 💻 Copy-paste Python examples
- 🚀 Common workflows
- 🐛 Debugging checklist
- ⏱️ Read time: 5-10 minutes

**Start with this if**: You want to train a model immediately

### ML_DATASET_DOCUMENTATION.md
- 📚 Complete documentation
- 🔍 Detailed feature descriptions
- 📊 Statistical analysis
- 💡 Model recommendations
- 🎯 Use case examples
- ⏱️ Read time: 30-60 minutes

**Start with this if**: You want to understand the data deeply

### GENERATION_SUMMARY.md
- 📦 What was created
- ✅ Quality checks
- 📈 Statistics
- 🎉 Summary
- ⏱️ Read time: 10-15 minutes

**Start with this if**: You want overview of outputs

### prepare_ml_dataset.py
- 🔧 Reproducible script
- 🛠️ Data preparation pipeline
- 🎛️ Customizable parameters
- 📝 Well-commented code
- ⏱️ Runtime: ~5 seconds

**Use this if**: You want to regenerate datasets or modify them

---

## 🚀 Getting Started Checklist

- [ ] Read `QUICK_START_GUIDE.md` (5 min)
- [ ] Load `crop_recommendation_complete.pkl` (1 min)
- [ ] Check `X_train.shape` and `y_train.shape` (1 min)
- [ ] Run example model from quick start (5 min)
- [ ] Check accuracy on test set (2 min)
- [ ] Try different dataset (crop_type_classification) (5 min)
- [ ] Read full `ML_DATASET_DOCUMENTATION.md` for details (30 min)

**Total Time to First Model**: ~20 minutes

---

## 📁 File Size Summary

| Category | File Count | Total Size |
|----------|-----------|-----------|
| Documentation | 3 | ~500 KB |
| Raw Data | 1 | ~15 KB |
| Processed CSV | 23 | ~200 KB |
| NPZ Archives | 5 | ~50 KB |
| Pickle Files | 6 | ~350 KB |
| JSON Files | 7 | ~100 KB |
| Python Script | 1 | ~35 KB |
| **TOTAL** | **36** | **~1.2 MB** |

---

## ✅ Verification Checklist

All files created and verified:

**Raw Data** ✓
- [ ] india_crops_complete.csv

**Documentation** ✓
- [ ] GENERATION_SUMMARY.md
- [ ] ML_DATASET_DOCUMENTATION.md
- [ ] QUICK_START_GUIDE.md

**Combined Datasets** ✓
- [ ] crops_ml_ready_full.csv
- [ ] crops_ml_numeric_only.csv
- [ ] crops_augmented.csv
- [ ] encoding_mappings.json

**5 Task-Specific Datasets** ✓
- [ ] crop_recommendation (5 files)
- [ ] crop_type_classification (5 files)
- [ ] growth_duration (5 files)
- [ ] water_requirement (5 files)
- [ ] season_classification (5 files)

**Encoders & Scalers** ✓
- [ ] label_encoders.json
- [ ] scalers.pkl

**Utility** ✓
- [ ] prepare_ml_dataset.py

---

## 🎯 Next Steps

1. **Read**: Start with `QUICK_START_GUIDE.md`
2. **Load**: Use provided code examples
3. **Train**: Try RandomForestClassifier
4. **Evaluate**: Check test set accuracy
5. **Explore**: Try different datasets
6. **Deep Dive**: Read full documentation
7. **Optimize**: Hyperparameter tuning
8. **Deploy**: Save model for production

---

## 📞 Support

- **Documentation**: See `ML_DATASET_DOCUMENTATION.md`
- **Quick Help**: See `QUICK_START_GUIDE.md`
- **Regenerate**: Run `prepare_ml_dataset.py`
- **Issues**: Check `QUICK_START_GUIDE.md` debugging section

---

**Status**: ✅ Complete and Ready to Use  
**Version**: 1.0  
**Created**: January 5, 2026  
**AgriSense ML Dataset** 🌾
