# Models Directory

This directory contains serialized ML model artifacts organized by group.

## Structure

```
models/
├── registry.json              # Central model manifest
├── group_a_tabular/          # Crop/Yield/Water/Fertilizer models
│   ├── catboost/
│   │   ├── crop_recommendation_v2.cbm
│   │   └── metadata.json
│   └── tfdf/
│       └── crop_recommendation_edge.tflite
├── group_b_vision/           # Disease/Weed detection models
│   ├── convnext/
│   │   └── disease_detector_v2.onnx
│   └── yolov8/
│       └── weed_segmentation_v2.onnx
├── group_c_edge/             # ESP32-S3 optimized models
│   └── crop_cnn1d_int8.tflite
├── group_d_nlp/              # Chatbot models
│   ├── distilbert_intent/
│   │   └── model.onnx
│   └── bge_m3/
│       └── embeddings.onnx
└── legacy/                   # Deprecated sklearn models
    ├── crop_recommendation_model.pkl
    └── ...
```

## Model Formats

| Format | Use Case | Size | Inference Speed |
|--------|----------|------|-----------------|
| `.cbm` | CatBoost native | Small | Fastest |
| `.onnx` | Cross-platform | Medium | Fast |
| `.tflite` | Edge/mobile | Smallest | Fast on edge |
| `.pt` | PyTorch (backup) | Large | GPU optimized |
| `.pkl` | Legacy sklearn | Small | Moderate |

## Adding New Models

1. Train model using scripts in `/training/`
2. Export to appropriate format
3. Register in `registry.json`
4. Update inference service to load new model
