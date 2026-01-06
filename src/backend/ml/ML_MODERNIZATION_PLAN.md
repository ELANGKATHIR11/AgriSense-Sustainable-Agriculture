# 🚀 AgriSense ML Architecture Modernization Plan

> **Version:** 2.0.0  
> **Date:** January 5, 2026  
> **Status:** Planning Phase

---

## 📊 Executive Summary

This document outlines the comprehensive modernization strategy for AgriSense's 18 ML models across 4 categories:
- **Group A**: Tabular Models (Crop/Yield/Water/Fertilizer) 
- **Group B**: Vision Models (Disease/Weed Detection)
- **Group C**: Edge/NPU Models (ESP32-S3 Targets)
- **Group D**: NLP & RAG (Multilingual Chatbot)

### Key Upgrades at a Glance

| Group | Current Stack | Upgraded Stack | Expected Gain |
|-------|--------------|----------------|---------------|
| **A** | RandomForest/GradientBoosting | CatBoost + TF-DF (Edge) | +8-12% accuracy |
| **B** | EfficientNet/YOLOv5 | ConvNeXt V2 Nano / YOLOv8-Seg | +15% mAP, real weed % |
| **C** | Dense MLP (TF) | 1D-CNN + QAT | 60% smaller, same accuracy |
| **D** | SVM + all-MiniLM-L6-v2 | DistilBERT + BGE-M3 | Hindi/Tamil support |

---

## 📁 Step 1: Reorganized Directory Structure

```
src/
└── backend/
    └── ml/
        │
        ├── __init__.py                 # Package exports
        ├── config/
        │   ├── __init__.py
        │   ├── model_registry.py       # Central model registration
        │   ├── training_config.yaml    # Hyperparameters & schedules
        │   └── inference_config.yaml   # Runtime configurations
        │
        ├── data/
        │   ├── __init__.py
        │   ├── loaders/
        │   │   ├── __init__.py
        │   │   ├── tabular_loader.py   # CSV/Parquet loaders
        │   │   ├── vision_loader.py    # Image dataset loaders
        │   │   └── nlp_loader.py       # Text corpus loaders
        │   ├── preprocessors/
        │   │   ├── __init__.py
        │   │   ├── tabular_preprocessor.py  # SMOTE-NC, Mixup, scaling
        │   │   ├── vision_preprocessor.py   # Copy-Paste, Mosaic aug
        │   │   └── nlp_preprocessor.py      # Tokenization, embeddings
        │   └── validators/
        │       ├── __init__.py
        │       └── schema_validator.py      # Pydantic data validation
        │
        ├── training/                   # ⚡ HEAVY TRAINING (Python only)
        │   ├── __init__.py
        │   │
        │   ├── group_a_tabular/        # 🌾 Crop/Yield/Water/Fertilizer
        │   │   ├── __init__.py
        │   │   ├── catboost_trainer.py        # CatBoost + DART mode
        │   │   ├── tfdf_trainer.py            # TensorFlow Decision Forests
        │   │   ├── augmentation.py            # SMOTE-NC, Mixup
        │   │   └── experiments/
        │   │       └── hyperparameter_search.py
        │   │
        │   ├── group_b_vision/         # 🔬 Disease/Weed Detection
        │   │   ├── __init__.py
        │   │   ├── convnext_disease_trainer.py    # ConvNeXt V2 Nano
        │   │   ├── yolov8_weed_trainer.py         # YOLOv8-Seg
        │   │   ├── augmentation.py                # Copy-Paste, Mosaic
        │   │   └── datasets/
        │   │       ├── plantvillage_adapter.py
        │   │       └── deepweeds_adapter.py
        │   │
        │   ├── group_c_edge/           # 📱 ESP32-S3 / NPU Models
        │   │   ├── __init__.py
        │   │   ├── cnn1d_crop_trainer.py      # 1D-CNN for tabular
        │   │   ├── qat_pipeline.py            # Quantization Aware Training
        │   │   ├── mobilenet_blocks.py        # MobileNetV3-Small blocks
        │   │   └── tflite_converter.py        # Export to TFLite Micro
        │   │
        │   ├── group_d_nlp/            # 💬 Chatbot / RAG
        │   │   ├── __init__.py
        │   │   ├── distilbert_intent.py       # Intent classification
        │   │   ├── bge_m3_embeddings.py       # BGE-M3 multilingual
        │   │   └── rag_pipeline.py            # Hybrid RAG
        │   │
        │   └── common/
        │       ├── __init__.py
        │       ├── callbacks.py               # Early stopping, checkpoints
        │       ├── metrics.py                 # Custom metrics (agri-specific)
        │       └── experiment_tracker.py      # MLflow/W&B integration
        │
        ├── inference/                  # 🚀 LIGHTWEIGHT INFERENCE
        │   ├── __init__.py
        │   │
        │   ├── engines/                # Runtime engines by format
        │   │   ├── __init__.py
        │   │   ├── onnx_engine.py             # ONNX Runtime
        │   │   ├── tflite_engine.py           # TFLite interpreter
        │   │   ├── catboost_engine.py         # CatBoost native
        │   │   └── torch_engine.py            # PyTorch (fallback)
        │   │
        │   ├── services/               # Business logic wrappers
        │   │   ├── __init__.py
        │   │   ├── crop_recommendation.py     # API-ready service
        │   │   ├── disease_detection.py       # Vision inference
        │   │   ├── weed_segmentation.py       # Instance segmentation
        │   │   ├── water_prediction.py        # Water requirement
        │   │   └── chatbot_service.py         # RAG + Intent
        │   │
        │   ├── optimizers/
        │   │   ├── __init__.py
        │   │   ├── batch_inference.py         # Batch processing
        │   │   └── model_warmer.py            # Pre-load models
        │   │
        │   └── cache/
        │       ├── __init__.py
        │       └── prediction_cache.py        # Redis/LRU cache
        │
        ├── models/                     # 📦 SERIALIZED ARTIFACTS
        │   ├── __init__.py
        │   ├── registry.json                  # Model manifest
        │   │
        │   ├── group_a_tabular/
        │   │   ├── catboost/
        │   │   │   ├── crop_recommendation_v2.cbm
        │   │   │   ├── yield_prediction_v2.cbm
        │   │   │   └── metadata.json
        │   │   └── tfdf/
        │   │       ├── crop_recommendation_edge.tflite
        │   │       └── metadata.json
        │   │
        │   ├── group_b_vision/
        │   │   ├── convnext/
        │   │   │   ├── disease_detector_v2.onnx
        │   │   │   └── metadata.json
        │   │   └── yolov8/
        │   │       ├── weed_segmentation_v2.onnx
        │   │       ├── weed_segmentation_v2.pt
        │   │       └── metadata.json
        │   │
        │   ├── group_c_edge/
        │   │   ├── crop_cnn1d_int8.tflite     # ESP32-S3 ready
        │   │   ├── soil_cnn1d_int8.tflite
        │   │   └── metadata.json
        │   │
        │   ├── group_d_nlp/
        │   │   ├── distilbert_intent/
        │   │   │   ├── model.onnx
        │   │   │   ├── tokenizer/
        │   │   │   └── metadata.json
        │   │   └── bge_m3/
        │   │       ├── embeddings.onnx
        │   │       └── metadata.json
        │   │
        │   └── legacy/                        # Old sklearn models (deprecated)
        │       ├── crop_recommendation_model.pkl
        │       └── ...
        │
        ├── edge/                       # 🔌 ESP32/EMBEDDED SPECIFIC
        │   ├── __init__.py
        │   ├── tflite_micro/
        │   │   ├── model_data.h               # C++ model headers
        │   │   └── inference_wrapper.cpp      # TFLite Micro wrapper
        │   ├── esp32_export/
        │   │   └── export_script.py           # Arduino/ESP-IDF export
        │   └── simulator/
        │       └── edge_simulator.py          # Test on desktop
        │
        ├── evaluation/                 # 📈 VALIDATION & BENCHMARKS
        │   ├── __init__.py
        │   ├── benchmarks/
        │   │   ├── accuracy_benchmark.py
        │   │   ├── latency_benchmark.py
        │   │   └── edge_benchmark.py          # ESP32 profiling
        │   ├── reports/
        │   │   └── model_card_generator.py    # Auto model cards
        │   └── tests/
        │       ├── test_tabular_models.py
        │       ├── test_vision_models.py
        │       └── test_edge_models.py
        │
        └── utils/
            ├── __init__.py
            ├── logging.py                     # Structured logging
            ├── device_utils.py                # GPU/NPU detection
            └── model_versioning.py            # Semantic versioning
```

---

## 📋 Step 2: Directory Purpose Summary

### `/training/` - Heavy Compute (Python-only)
- **Purpose**: GPU/CPU intensive training scripts
- **Dependencies**: Full PyTorch, TensorFlow, CatBoost, Ultralytics
- **Runs on**: Training server (cloud GPU / local workstation)
- **Artifacts**: Produces checkpoints, then exports to `/models/`

### `/inference/` - Lightweight Runtime
- **Purpose**: Fast prediction services for FastAPI
- **Dependencies**: ONNX Runtime, TFLite, minimal PyTorch
- **Runs on**: Production server / edge devices
- **Loads from**: `/models/` (ONNX, TFLite, CatBoost native)

### `/edge/` - Embedded Device Code
- **Purpose**: ESP32-S3 / Arduino deployment
- **Dependencies**: None (C/C++ headers generated from Python)
- **Runs on**: Microcontrollers (TFLite Micro)

### `/models/` - Serialized Artifacts
- **Purpose**: Version-controlled model storage
- **Formats**: 
  - `.cbm` - CatBoost models
  - `.onnx` - Cross-platform neural networks
  - `.tflite` - TensorFlow Lite (quantized)
  - `.pt` - PyTorch checkpoints (backup)

---

## 🗓️ Migration Path

### Phase 1: Foundation (Week 1-2)
1. ✅ Create new directory structure
2. ✅ Set up requirements files
3. ⏳ Migrate existing models to `/models/legacy/`

### Phase 2: Group A - Tabular (Week 3-4)
1. Implement CatBoost trainer with DART
2. Add SMOTE-NC data augmentation
3. Export TF-DF for edge

### Phase 3: Group C - Edge (Week 4-5)
1. Build 1D-CNN with MobileNetV3 blocks
2. Implement QAT pipeline
3. Test on ESP32-S3 simulator

### Phase 4: Group B - Vision (Week 5-7)
1. Train ConvNeXt V2 Nano on PlantVillage
2. Fine-tune YOLOv8-Seg on DeepWeeds
3. Implement Copy-Paste augmentation

### Phase 5: Group D - NLP (Week 7-8)
1. Fine-tune DistilBERT for intent
2. Integrate BGE-M3 embeddings
3. Update RAG pipeline

---

## ✅ Next Steps

Once you confirm this structure:
1. I will generate the optimized `requirements.txt` (Step 2)
2. Then await your confirmation for code generation (Steps 3-4)

**Confirm to proceed?**
