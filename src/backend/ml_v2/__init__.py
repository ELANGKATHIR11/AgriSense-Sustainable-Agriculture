"""
AgriSense ML Package - Modernized Architecture v2.0

This package contains the modernized ML pipeline for precision agriculture:

Groups:
    - group_a_tabular: CatBoost + TF-DF for crop/yield/water/fertilizer
    - group_b_vision: ConvNeXt V2 + YOLOv8-Seg for disease/weed
    - group_c_edge: 1D-CNN with QAT for ESP32-S3
    - group_d_nlp: DistilBERT + BGE-M3 for multilingual chatbot

Usage:
    # Inference (lightweight)
    from ml.inference.services import CropRecommendationService
    service = CropRecommendationService()
    result = service.predict(features)
    
    # Training (heavy)
    from ml.training.group_a_tabular import CatBoostTrainer
    trainer = CatBoostTrainer(config_path="config/training_config.yaml")
    trainer.train()
"""

__version__ = "2.0.0"
__author__ = "AgriSense Team"

# Lazy imports to avoid loading heavy deps on inference-only installs
def get_training_module(group: str):
    """Dynamically import training modules."""
    if group == "tabular":
        from .training import group_a_tabular
        return group_a_tabular
    elif group == "vision":
        from .training import group_b_vision
        return group_b_vision
    elif group == "edge":
        from .training import group_c_edge
        return group_c_edge
    elif group == "nlp":
        from .training import group_d_nlp
        return group_d_nlp
    else:
        raise ValueError(f"Unknown group: {group}")
