"""
ML Configuration Module

Contains:
    - model_registry: Central model registration and versioning
    - training_config.yaml: Hyperparameters and schedules
    - inference_config.yaml: Runtime configurations
"""

from .model_registry import ModelRegistry

__all__ = ["ModelRegistry"]
