"""
Model Registry - Central model registration and versioning system.

Tracks all trained models with metadata, performance metrics, and deployment status.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    VALIDATING = "validating"
    STAGED = "staged"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelFormat(str, Enum):
    """Supported model formats."""
    CATBOOST = "cbm"
    ONNX = "onnx"
    TFLITE = "tflite"
    PYTORCH = "pt"
    PICKLE = "pkl"  # Legacy


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    name: str
    version: str
    group: str  # group_a, group_b, group_c, group_d
    format: ModelFormat
    status: ModelStatus
    
    # Performance metrics
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    latency_ms: Optional[float] = None
    model_size_mb: Optional[float] = None
    
    # Training info
    trained_at: Optional[str] = None
    training_dataset: Optional[str] = None
    training_samples: Optional[int] = None
    
    # Deployment info
    deployed_at: Optional[str] = None
    deployment_target: Optional[str] = None  # server, edge, both
    
    # Paths
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    
    # Additional metadata
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ModelRegistry:
    """
    Central registry for all ML models.
    
    Usage:
        registry = ModelRegistry()
        
        # Register a new model
        registry.register(ModelMetadata(
            model_id="crop_rec_v2",
            name="Crop Recommendation CatBoost",
            version="2.0.0",
            group="group_a",
            format=ModelFormat.CATBOOST,
            status=ModelStatus.STAGED,
            accuracy=0.94
        ))
        
        # Get production model
        model = registry.get_production_model("crop_recommendation")
        
        # List all models by group
        tabular_models = registry.list_models(group="group_a")
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize registry from JSON file."""
        self.registry_path = registry_path or Path(__file__).parent.parent / "models" / "registry.json"
        self._models: Dict[str, ModelMetadata] = {}
        self._load()
    
    def _load(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                for model_id, model_data in data.get("models", {}).items():
                    # Convert string enums back to Enum
                    model_data["format"] = ModelFormat(model_data["format"])
                    model_data["status"] = ModelStatus(model_data["status"])
                    self._models[model_id] = ModelMetadata(**model_data)
    
    def _save(self):
        """Persist registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "2.0",
            "updated_at": datetime.utcnow().isoformat(),
            "models": {
                model_id: {
                    **model.to_dict(),
                    "format": model.format.value,
                    "status": model.status.value
                }
                for model_id, model in self._models.items()
            }
        }
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register(self, metadata: ModelMetadata) -> None:
        """Register a new model or update existing."""
        self._models[metadata.model_id] = metadata
        self._save()
    
    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model by ID."""
        return self._models.get(model_id)
    
    def get_production_model(self, model_name: str) -> Optional[ModelMetadata]:
        """Get the production version of a model by name."""
        for model in self._models.values():
            if model.name == model_name and model.status == ModelStatus.PRODUCTION:
                return model
        return None
    
    def list_models(
        self, 
        group: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        format: Optional[ModelFormat] = None
    ) -> List[ModelMetadata]:
        """List models with optional filtering."""
        models = list(self._models.values())
        
        if group:
            models = [m for m in models if m.group == group]
        if status:
            models = [m for m in models if m.status == status]
        if format:
            models = [m for m in models if m.format == format]
        
        return models
    
    def promote_to_production(self, model_id: str) -> None:
        """Promote a staged model to production."""
        model = self._models.get(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        if model.status != ModelStatus.STAGED:
            raise ValueError(f"Model must be staged before promotion: {model.status}")
        
        # Deprecate current production model of same name
        for m in self._models.values():
            if m.name == model.name and m.status == ModelStatus.PRODUCTION:
                m.status = ModelStatus.DEPRECATED
        
        model.status = ModelStatus.PRODUCTION
        model.deployed_at = datetime.utcnow().isoformat()
        self._save()
    
    def deprecate(self, model_id: str) -> None:
        """Mark model as deprecated."""
        model = self._models.get(model_id)
        if model:
            model.status = ModelStatus.DEPRECATED
            self._save()
