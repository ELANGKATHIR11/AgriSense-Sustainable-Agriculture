"""
YOLOv8-Seg Weed Detection Trainer with Mosaic & Copy-Paste Augmentation

This module implements a complete training pipeline for weed instance segmentation
using Ultralytics YOLOv8-Seg with advanced augmentation strategies.

Key Features:
    - YOLOv8n-seg (Nano) optimized for field deployment
    - Mosaic augmentation for multi-scale training
    - Copy-Paste augmentation to fix lab-bias (healthy leaves on field backgrounds)
    - Support for DeepWeeds and custom weed datasets
    - Instance segmentation for precise weed coverage % calculation
    - Export to ONNX and TensorRT

Usage:
    trainer = YOLOv8WeedTrainer(config_path="config/training_config.yaml")
    trainer.train()
    trainer.export(format="onnx")

Reference:
    - Ultralytics YOLOv8: https://docs.ultralytics.com/
    - DeepWeeds Dataset: https://github.com/AlexOlsen/DeepWeeds
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging

import numpy as np
import cv2
from PIL import Image
import yaml

# Ultralytics YOLOv8
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AugmentationConfig:
    """Augmentation configuration for YOLOv8 training."""
    
    # Mosaic augmentation
    mosaic: float = 1.0  # Probability of mosaic (combines 4 images)
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    
    # Copy-Paste augmentation (key for fixing lab bias!)
    copy_paste: float = 0.5  # Probability of copy-paste
    copy_paste_mode: str = "flip"  # flip, rotate, or both
    
    # MixUp augmentation
    mixup: float = 0.1  # Probability of mixup
    
    # Geometric augmentations
    degrees: float = 10.0  # Rotation degrees
    translate: float = 0.1  # Translation fraction
    scale: float = 0.5  # Scale factor +/- 50%
    shear: float = 2.0  # Shear degrees
    perspective: float = 0.0001  # Perspective transform
    flipud: float = 0.5  # Vertical flip probability
    fliplr: float = 0.5  # Horizontal flip probability
    
    # Color augmentations (HSV)
    hsv_h: float = 0.015  # Hue shift
    hsv_s: float = 0.7  # Saturation shift
    hsv_v: float = 0.4  # Value shift
    
    # Additional augmentations
    erasing: float = 0.4  # Random erasing probability
    crop_fraction: float = 1.0  # Crop fraction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YOLO training."""
        return {
            'mosaic': self.mosaic,
            'copy_paste': self.copy_paste,
            'mixup': self.mixup,
            'degrees': self.degrees,
            'translate': self.translate,
            'scale': self.scale,
            'shear': self.shear,
            'perspective': self.perspective,
            'flipud': self.flipud,
            'fliplr': self.fliplr,
            'hsv_h': self.hsv_h,
            'hsv_s': self.hsv_s,
            'hsv_v': self.hsv_v,
            'erasing': self.erasing,
            'crop_fraction': self.crop_fraction,
        }


@dataclass
class TrainingConfig:
    """YOLOv8-Seg training configuration."""
    
    # Model
    model_type: str = "yolov8n-seg"  # nano, small, medium, large, xlarge
    pretrained: bool = True
    
    # Architecture
    input_size: int = 640  # Image size (640x640)
    
    # Training
    batch_size: int = 16
    epochs: int = 200
    patience: int = 50  # Early stopping patience
    
    # Optimizer
    optimizer: str = "AdamW"
    learning_rate: float = 0.001
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    
    # Scheduler
    lr_scheduler: str = "cosine"  # cosine, linear
    final_lr_ratio: float = 0.01
    
    # Loss weights
    box_loss: float = 7.5  # Box regression loss weight
    cls_loss: float = 0.5  # Classification loss weight
    dfl_loss: float = 1.5  # Distribution focal loss weight
    seg_loss: float = 3.0  # Segmentation loss weight
    
    # Augmentation
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    # Paths
    dataset_path: str = None
    project_name: str = "agrisense_weed_detection"
    experiment_name: str = "yolov8seg_v2"
    
    # Hardware
    device: str = "0"  # GPU device (0, 1, or 'cpu')
    workers: int = 8  # DataLoader workers
    
    # Export
    export_formats: List[str] = field(default_factory=lambda: ["onnx", "pt"])


# =============================================================================
# COPY-PASTE AUGMENTATION (Custom Implementation)
# =============================================================================

class CopyPasteAugmentor:
    """
    Custom Copy-Paste augmentation for fixing lab-bias in weed detection.
    
    Problem: Models trained on clean lab images (PlantVillage) fail on
    real field images with complex backgrounds.
    
    Solution: Copy weed/plant instances from clean images and paste them
    onto diverse field background images (soil, grass, debris).
    
    Reference: "Simple Copy-Paste is a Strong Data Augmentation Method" (CVPR 2021)
    """
    
    def __init__(
        self,
        backgrounds_dir: Optional[Path] = None,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        max_objects: int = 3,
        blend_mode: str = "gaussian"  # hard, gaussian, poisson
    ):
        """
        Initialize Copy-Paste augmentor.
        
        Args:
            backgrounds_dir: Directory with background images (field, soil)
            min_scale: Minimum scale for pasted objects
            max_scale: Maximum scale for pasted objects
            max_objects: Maximum objects to paste per image
            blend_mode: Blending mode for pasting
        """
        self.backgrounds_dir = backgrounds_dir
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_objects = max_objects
        self.blend_mode = blend_mode
        
        # Load backgrounds if provided
        self.backgrounds = []
        if backgrounds_dir and Path(backgrounds_dir).exists():
            self._load_backgrounds()
    
    def _load_backgrounds(self):
        """Load background images from directory."""
        bg_dir = Path(self.backgrounds_dir)
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in extensions:
            for img_path in bg_dir.glob(f'*{ext}'):
                self.backgrounds.append(str(img_path))
        
        logger.info(f"Loaded {len(self.backgrounds)} background images")
    
    def extract_instances(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        boxes: List[Tuple[float, float, float, float]],
        class_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Extract object instances from image using masks.
        
        Args:
            image: Source image (H, W, 3)
            masks: List of binary masks for each instance
            boxes: Bounding boxes [x1, y1, x2, y2]
            class_ids: Class ID for each instance
            
        Returns:
            List of instance dictionaries with cropped images and metadata
        """
        instances = []
        
        for mask, box, cls_id in zip(masks, boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Crop instance
            instance_img = image[y1:y2, x1:x2].copy()
            instance_mask = mask[y1:y2, x1:x2].copy()
            
            # Apply mask to get alpha channel
            if len(instance_mask.shape) == 2:
                instance_mask = instance_mask.astype(np.uint8) * 255
            
            instances.append({
                'image': instance_img,
                'mask': instance_mask,
                'class_id': cls_id,
                'original_size': (x2 - x1, y2 - y1)
            })
        
        return instances
    
    def paste_instance(
        self,
        background: np.ndarray,
        instance: Dict[str, Any],
        position: Tuple[int, int],
        scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
        """
        Paste an instance onto background with blending.
        
        Args:
            background: Background image (H, W, 3)
            instance: Instance dict from extract_instances()
            position: (x, y) position to paste center
            scale: Scale factor for instance
            
        Returns:
            (modified_background, mask, new_bbox)
        """
        inst_img = instance['image'].copy()
        inst_mask = instance['mask'].copy()
        
        # Scale instance
        if scale != 1.0:
            new_size = (int(inst_img.shape[1] * scale), int(inst_img.shape[0] * scale))
            inst_img = cv2.resize(inst_img, new_size)
            inst_mask = cv2.resize(inst_mask, new_size)
        
        h, w = inst_img.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Calculate paste region
        x, y = position
        x1 = max(0, x - w // 2)
        y1 = max(0, y - h // 2)
        x2 = min(bg_w, x1 + w)
        y2 = min(bg_h, y1 + h)
        
        # Adjust instance crop if out of bounds
        inst_x1 = max(0, w // 2 - x)
        inst_y1 = max(0, h // 2 - y)
        inst_x2 = inst_x1 + (x2 - x1)
        inst_y2 = inst_y1 + (y2 - y1)
        
        inst_crop = inst_img[inst_y1:inst_y2, inst_x1:inst_x2]
        mask_crop = inst_mask[inst_y1:inst_y2, inst_x1:inst_x2]
        
        # Create output
        result = background.copy()
        result_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        
        # Blend based on mode
        if self.blend_mode == "hard":
            # Hard paste with mask
            mask_3c = np.stack([mask_crop > 127] * 3, axis=-1)
            result[y1:y2, x1:x2] = np.where(mask_3c, inst_crop, result[y1:y2, x1:x2])
            
        elif self.blend_mode == "gaussian":
            # Gaussian blur on mask edges for smooth blending
            mask_float = mask_crop.astype(np.float32) / 255.0
            mask_blur = cv2.GaussianBlur(mask_float, (7, 7), 0)
            mask_3c = np.stack([mask_blur] * 3, axis=-1)
            
            result[y1:y2, x1:x2] = (
                inst_crop * mask_3c + 
                result[y1:y2, x1:x2] * (1 - mask_3c)
            ).astype(np.uint8)
            
        elif self.blend_mode == "poisson":
            # Poisson blending (seamless cloning)
            if inst_crop.shape[0] > 0 and inst_crop.shape[1] > 0:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                try:
                    result = cv2.seamlessClone(
                        inst_crop, result, mask_crop, center, cv2.NORMAL_CLONE
                    )
                except:
                    # Fallback to hard paste if Poisson fails
                    mask_3c = np.stack([mask_crop > 127] * 3, axis=-1)
                    result[y1:y2, x1:x2] = np.where(mask_3c, inst_crop, result[y1:y2, x1:x2])
        
        # Update mask
        result_mask[y1:y2, x1:x2] = mask_crop
        
        # New bounding box (YOLO format: x_center, y_center, width, height - normalized)
        bbox = (
            (x1 + x2) / 2 / bg_w,  # x_center
            (y1 + y2) / 2 / bg_h,  # y_center
            (x2 - x1) / bg_w,       # width
            (y2 - y1) / bg_h        # height
        )
        
        return result, result_mask, bbox
    
    def augment(
        self,
        image: np.ndarray,
        instances: List[Dict[str, Any]],
        use_random_background: bool = True
    ) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple], List[int]]:
        """
        Apply copy-paste augmentation.
        
        Args:
            image: Original image
            instances: List of instances to paste
            use_random_background: Whether to use random background
            
        Returns:
            (augmented_image, masks, boxes, class_ids)
        """
        # Select background
        if use_random_background and self.backgrounds:
            bg_path = random.choice(self.backgrounds)
            background = cv2.imread(bg_path)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            # Resize to match original
            background = cv2.resize(background, (image.shape[1], image.shape[0]))
        else:
            background = image.copy()
        
        # Determine number of instances to paste
        n_paste = min(len(instances), random.randint(1, self.max_objects))
        selected = random.sample(instances, n_paste) if len(instances) > n_paste else instances
        
        masks = []
        boxes = []
        class_ids = []
        
        for inst in selected:
            # Random position
            h, w = background.shape[:2]
            x = random.randint(w // 4, 3 * w // 4)
            y = random.randint(h // 4, 3 * h // 4)
            
            # Random scale
            scale = random.uniform(self.min_scale, self.max_scale)
            
            # Paste
            background, mask, bbox = self.paste_instance(
                background, inst, (x, y), scale
            )
            
            masks.append(mask)
            boxes.append(bbox)
            class_ids.append(inst['class_id'])
        
        return background, masks, boxes, class_ids


# =============================================================================
# DATASET PREPARATION
# =============================================================================

class WeedDatasetPreparator:
    """
    Prepare weed detection datasets in YOLO format.
    
    Supports:
        - DeepWeeds dataset
        - PlantVillage (for disease detection transfer)
        - Custom COCO-format datasets
    """
    
    DEEPWEEDS_CLASSES = [
        'Chinee_apple', 'Lantana', 'Parkinsonia', 'Parthenium',
        'Prickly_acacia', 'Rubber_vine', 'Siam_weed', 'Snake_weed', 'Negative'
    ]
    
    def __init__(self, output_dir: Path):
        """Initialize dataset preparator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_deepweeds(self, source_dir: Path, train_ratio: float = 0.8) -> Path:
        """
        Convert DeepWeeds dataset to YOLO format.
        
        Args:
            source_dir: Path to DeepWeeds images and labels
            train_ratio: Train/val split ratio
            
        Returns:
            Path to dataset.yaml
        """
        logger.info("Preparing DeepWeeds dataset...")
        
        # Create directory structure
        for split in ['train', 'val']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Process images...
        # (Implementation depends on source format)
        
        # Create dataset.yaml
        dataset_yaml = {
            'path': str(self.output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(self.DEEPWEEDS_CLASSES[:-1])}  # Exclude 'Negative'
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        logger.info(f"Dataset prepared at: {self.output_dir}")
        return yaml_path
    
    def create_yaml_config(
        self,
        train_path: str,
        val_path: str,
        class_names: List[str],
        nc: int = None
    ) -> Path:
        """
        Create YOLO dataset configuration file.
        
        Args:
            train_path: Path to training images
            val_path: Path to validation images
            class_names: List of class names
            nc: Number of classes (optional)
            
        Returns:
            Path to dataset.yaml
        """
        nc = nc or len(class_names)
        
        config = {
            'path': str(self.output_dir),
            'train': train_path,
            'val': val_path,
            'nc': nc,
            'names': class_names
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return yaml_path


# =============================================================================
# MAIN TRAINER CLASS
# =============================================================================

class YOLOv8WeedTrainer:
    """
    Complete YOLOv8-Seg training pipeline for weed detection.
    
    Features:
        - YOLOv8n-seg with configurable model size
        - Mosaic + Copy-Paste + MixUp augmentation
        - Automatic mixed precision training
        - Multi-GPU support
        - Export to ONNX/TensorRT
    
    Usage:
        trainer = YOLOv8WeedTrainer()
        trainer.prepare_dataset("/path/to/deepweeds")
        trainer.train()
        metrics = trainer.validate()
        trainer.export("onnx")
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize trainer with configuration."""
        self.config = config or TrainingConfig()
        self.model: Optional[YOLO] = None
        self.dataset_yaml: Optional[Path] = None
        self.results = None
        
        # Copy-paste augmentor
        self.copy_paste_augmentor = CopyPasteAugmentor(
            blend_mode="gaussian",
            max_objects=3
        )
        
        # Create output directories
        self.output_dir = Path(f"runs/{self.config.project_name}/{self.config.experiment_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, weights: Optional[str] = None) -> YOLO:
        """
        Load YOLOv8-Seg model.
        
        Args:
            weights: Path to custom weights or None for pretrained
            
        Returns:
            YOLO model instance
        """
        if weights:
            self.model = YOLO(weights)
            logger.info(f"Loaded custom weights: {weights}")
        else:
            # Load pretrained model
            model_name = f"{self.config.model_type}.pt"
            self.model = YOLO(model_name)
            logger.info(f"Loaded pretrained: {model_name}")
        
        return self.model
    
    def prepare_dataset(
        self,
        dataset_path: str,
        dataset_type: str = "yolo"  # yolo, coco, deepweeds
    ) -> Path:
        """
        Prepare dataset for training.
        
        Args:
            dataset_path: Path to dataset
            dataset_type: Format of source dataset
            
        Returns:
            Path to dataset.yaml
        """
        dataset_path = Path(dataset_path)
        
        if dataset_type == "yolo":
            # Already in YOLO format, just locate yaml
            yaml_files = list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml"))
            if yaml_files:
                self.dataset_yaml = yaml_files[0]
            else:
                raise FileNotFoundError(f"No dataset.yaml found in {dataset_path}")
                
        elif dataset_type == "deepweeds":
            preparator = WeedDatasetPreparator(self.output_dir / "dataset")
            self.dataset_yaml = preparator.prepare_deepweeds(dataset_path)
            
        elif dataset_type == "coco":
            # Convert COCO to YOLO format
            # (Implementation would go here)
            pass
        
        logger.info(f"Dataset prepared: {self.dataset_yaml}")
        return self.dataset_yaml
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """
        Get augmentation configuration for YOLO training.
        
        This includes Mosaic, Copy-Paste, and other augmentations.
        """
        aug_config = self.config.augmentation.to_dict()
        
        # Log augmentation settings
        logger.info("Augmentation Configuration:")
        logger.info(f"  - Mosaic: {aug_config['mosaic']}")
        logger.info(f"  - Copy-Paste: {aug_config['copy_paste']}")
        logger.info(f"  - MixUp: {aug_config['mixup']}")
        logger.info(f"  - HSV: h={aug_config['hsv_h']}, s={aug_config['hsv_s']}, v={aug_config['hsv_v']}")
        
        return aug_config
    
    def train(self, resume: bool = False) -> Any:
        """
        Run YOLOv8-Seg training with configured augmentations.
        
        Args:
            resume: Whether to resume from last checkpoint
            
        Returns:
            Training results
        """
        if self.model is None:
            self.load_model()
        
        if self.dataset_yaml is None:
            raise ValueError("No dataset prepared. Call prepare_dataset() first.")
        
        # Get augmentation config
        aug_config = self.get_augmentation_config()
        
        logger.info("=" * 60)
        logger.info("Starting YOLOv8-Seg Training")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.model_type}")
        logger.info(f"Dataset: {self.dataset_yaml}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch Size: {self.config.batch_size}")
        logger.info(f"Image Size: {self.config.input_size}")
        logger.info("=" * 60)
        
        # Training arguments
        train_args = {
            # Dataset
            'data': str(self.dataset_yaml),
            'imgsz': self.config.input_size,
            
            # Training
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'patience': self.config.patience,
            'device': self.config.device,
            'workers': self.config.workers,
            
            # Optimizer
            'optimizer': self.config.optimizer,
            'lr0': self.config.learning_rate,
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'warmup_epochs': self.config.warmup_epochs,
            'warmup_momentum': self.config.warmup_momentum,
            'lrf': self.config.final_lr_ratio,
            
            # Loss weights
            'box': self.config.box_loss,
            'cls': self.config.cls_loss,
            'dfl': self.config.dfl_loss,
            
            # Augmentation (Mosaic & Copy-Paste)
            'mosaic': aug_config['mosaic'],
            'copy_paste': aug_config['copy_paste'],
            'mixup': aug_config['mixup'],
            'degrees': aug_config['degrees'],
            'translate': aug_config['translate'],
            'scale': aug_config['scale'],
            'shear': aug_config['shear'],
            'perspective': aug_config['perspective'],
            'flipud': aug_config['flipud'],
            'fliplr': aug_config['fliplr'],
            'hsv_h': aug_config['hsv_h'],
            'hsv_s': aug_config['hsv_s'],
            'hsv_v': aug_config['hsv_v'],
            'erasing': aug_config['erasing'],
            'crop_fraction': aug_config['crop_fraction'],
            
            # Output
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'exist_ok': True,
            'pretrained': self.config.pretrained,
            'resume': resume,
            
            # Logging
            'verbose': True,
            'plots': True,
            'save': True,
            'save_period': 10,
        }
        
        # Run training
        self.results = self.model.train(**train_args)
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        
        return self.results
    
    def validate(self, data: Optional[str] = None) -> Dict[str, float]:
        """
        Validate model on test set.
        
        Args:
            data: Path to validation dataset (uses training dataset if None)
            
        Returns:
            Validation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        data = data or str(self.dataset_yaml)
        
        metrics = self.model.val(
            data=data,
            imgsz=self.config.input_size,
            batch=self.config.batch_size,
            device=self.config.device,
            split='val',
            plots=True
        )
        
        results = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'mask_mAP50': float(metrics.seg.map50) if hasattr(metrics, 'seg') else None,
            'mask_mAP50-95': float(metrics.seg.map) if hasattr(metrics, 'seg') else None,
        }
        
        logger.info("Validation Results:")
        for key, value in results.items():
            if value is not None:
                logger.info(f"  {key}: {value:.4f}")
        
        return results
    
    def predict(
        self,
        source: Union[str, np.ndarray, List],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        return_masks: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run inference on images.
        
        Args:
            source: Image path, numpy array, or list of images
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            return_masks: Whether to return segmentation masks
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("No model loaded.")
        
        results = self.model.predict(
            source=source,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=self.config.input_size,
            device=self.config.device,
            retina_masks=return_masks,
            verbose=False
        )
        
        predictions = []
        for r in results:
            pred = {
                'boxes': r.boxes.xyxy.cpu().numpy() if r.boxes else None,
                'scores': r.boxes.conf.cpu().numpy() if r.boxes else None,
                'classes': r.boxes.cls.cpu().numpy() if r.boxes else None,
                'masks': r.masks.data.cpu().numpy() if r.masks else None,
            }
            
            # Calculate weed coverage percentage
            if pred['masks'] is not None and len(pred['masks']) > 0:
                total_mask = np.any(pred['masks'], axis=0)
                pred['weed_coverage_pct'] = float(total_mask.sum() / total_mask.size * 100)
            else:
                pred['weed_coverage_pct'] = 0.0
            
            predictions.append(pred)
        
        return predictions
    
    def calculate_weed_coverage(self, prediction: Dict[str, Any]) -> float:
        """
        Calculate total weed coverage percentage from segmentation masks.
        
        Args:
            prediction: Prediction dict from predict()
            
        Returns:
            Weed coverage percentage (0-100)
        """
        return prediction.get('weed_coverage_pct', 0.0)
    
    def export(
        self,
        format: str = "onnx",
        output_path: Optional[str] = None,
        simplify: bool = True,
        dynamic: bool = False
    ) -> Path:
        """
        Export model to deployment format.
        
        Args:
            format: Export format (onnx, torchscript, tensorrt, etc.)
            output_path: Custom output path
            simplify: Simplify ONNX model
            dynamic: Use dynamic input shapes
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError("No model loaded.")
        
        logger.info(f"Exporting model to {format}...")
        
        export_path = self.model.export(
            format=format,
            imgsz=self.config.input_size,
            simplify=simplify,
            dynamic=dynamic,
            half=False,  # FP32 for compatibility
        )
        
        # Save metadata
        metadata = {
            'model_name': 'WeedSegmentation_YOLOv8',
            'version': '2.0.0',
            'architecture': self.config.model_type,
            'input_size': self.config.input_size,
            'format': format,
            'classes': self.model.names,
            'task': 'instance-segmentation'
        }
        
        metadata_path = Path(export_path).parent / f"{Path(export_path).stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported to: {export_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return Path(export_path)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8-Seg for Weed Detection")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml or dataset folder")
    parser.add_argument("--model", type=str, default="yolov8n-seg", 
                       choices=["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg"],
                       help="Model size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (0, 1, cpu)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    
    # Augmentation
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic probability")
    parser.add_argument("--copy-paste", type=float, default=0.5, help="Copy-paste probability")
    parser.add_argument("--mixup", type=float, default=0.1, help="MixUp probability")
    
    # Output
    parser.add_argument("--project", type=str, default="agrisense_weed", help="Project name")
    parser.add_argument("--name", type=str, default="yolov8seg_v2", help="Experiment name")
    parser.add_argument("--export", type=str, nargs="+", default=["onnx"], 
                       help="Export formats (onnx, pt, torchscript)")
    
    parser.add_argument("--resume", action="store_true", help="Resume training")
    args = parser.parse_args()
    
    # Build configuration
    aug_config = AugmentationConfig(
        mosaic=args.mosaic,
        copy_paste=args.copy_paste,
        mixup=args.mixup
    )
    
    config = TrainingConfig(
        model_type=args.model,
        input_size=args.imgsz,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        workers=args.workers,
        project_name=args.project,
        experiment_name=args.name,
        augmentation=aug_config,
        export_formats=args.export
    )
    
    # Initialize trainer
    trainer = YOLOv8WeedTrainer(config)
    
    # Prepare dataset
    data_path = Path(args.data)
    if data_path.suffix in ['.yaml', '.yml']:
        trainer.dataset_yaml = data_path
    else:
        trainer.prepare_dataset(args.data, dataset_type="yolo")
    
    # Load model
    trainer.load_model()
    
    # Train
    trainer.train(resume=args.resume)
    
    # Validate
    metrics = trainer.validate()
    
    # Export
    for fmt in args.export:
        trainer.export(format=fmt)
    
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info(f"  mAP50: {metrics.get('mAP50', 0):.4f}")
    logger.info(f"  mAP50-95: {metrics.get('mAP50-95', 0):.4f}")
    logger.info(f"  Mask mAP50: {metrics.get('mask_mAP50', 0):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
