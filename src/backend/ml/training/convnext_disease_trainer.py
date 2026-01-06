#!/usr/bin/env python3
"""
AgriSense ConvNeXt V2 Disease Detection Trainer
================================================
Trains ConvNeXt V2 Nano for plant disease classification.

Features:
- ConvNeXt V2 Nano backbone from timm
- Copy-Paste augmentation integration
- Progressive resizing training
- Label smoothing and Mixup
- ONNX export for inference

Usage:
    python convnext_disease_trainer.py --data-dir ../data/images/diseases

Author: AgriSense ML Team
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
import random

import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed")

try:
    import timm
    from timm.data import create_transform
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    from timm.scheduler import CosineLRScheduler
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not installed. Install with: pip install timm")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.warning("albumentations not installed")

try:
    from torchvision import transforms, datasets
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class DiseaseDataset(Dataset):
    """
    Custom dataset for plant disease images.
    
    Supports:
    - Standard ImageFolder structure
    - On-the-fly augmentation
    - Copy-paste augmentation
    """
    
    def __init__(self, 
                 root_dir: str,
                 transform=None,
                 copy_paste_prob: float = 0.0,
                 background_dir: str = None):
        """
        Args:
            root_dir: Root directory with class subfolders
            transform: Image transforms
            copy_paste_prob: Probability of applying copy-paste augmentation
            background_dir: Directory with field backgrounds
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.copy_paste_prob = copy_paste_prob
        self.background_dir = Path(background_dir) if background_dir else None
        
        # Load images and labels
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        self._load_dataset()
        
        # Load backgrounds if available
        self.backgrounds = []
        if self.background_dir and self.background_dir.exists():
            self.backgrounds = list(self.background_dir.glob('*.jpg'))
    
    def _load_dataset(self):
        """Load dataset from directory structure."""
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx
            
            # Get all images in class directory
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), idx))
        
        logger.info(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply copy-paste augmentation
        if self.backgrounds and random.random() < self.copy_paste_prob:
            image = self._apply_copy_paste(image)
        
        # Apply transforms
        if self.transform:
            if ALBUMENTATIONS_AVAILABLE and isinstance(self.transform, A.Compose):
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = Image.fromarray(image)
                image = self.transform(image)
        
        return image, label
    
    def _apply_copy_paste(self, image: np.ndarray) -> np.ndarray:
        """Apply copy-paste augmentation."""
        if not self.backgrounds:
            return image
        
        # Load random background
        bg_path = random.choice(self.backgrounds)
        background = np.array(Image.open(bg_path).convert('RGB'))
        
        # Resize background to match image
        h, w = image.shape[:2]
        background = np.array(Image.fromarray(background).resize((w, h)))
        
        # Simple threshold-based background removal
        # (assumes lab background is darker/uniform)
        gray = np.mean(image, axis=2)
        mask = gray > 30  # Threshold for non-black pixels
        
        # Expand mask dimensions
        mask = np.stack([mask] * 3, axis=2)
        
        # Composite
        result = np.where(mask, image, background)
        
        return result.astype(np.uint8)


class ConvNeXtDiseaseClassifier(nn.Module):
    """
    ConvNeXt V2 Nano classifier for plant disease detection.
    """
    
    def __init__(self, 
                 num_classes: int,
                 pretrained: bool = True,
                 drop_rate: float = 0.1):
        super().__init__()
        
        # Load ConvNeXt V2 Nano backbone
        self.backbone = timm.create_model(
            'convnextv2_nano.fcmae_ft_in22k_in1k_384',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            drop_rate=drop_rate
        )
        
        # Get feature dimension
        self.num_features = self.backbone.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Dropout(drop_rate),
            nn.Linear(self.num_features, num_classes)
        )
        
        logger.info(f"ConvNeXt V2 Nano initialized with {num_classes} classes")
        logger.info(f"Feature dimension: {self.num_features}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings."""
        return self.backbone(x)


class Mixup:
    """Mixup data augmentation for classification."""
    
    def __init__(self, alpha: float = 0.8, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if random.random() > self.prob:
            return x, y, y, 1.0
        
        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        indices = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[indices]
        y_a, y_b = y, y[indices]
        
        return mixed_x, y_a, y_b, lam


class ConvNeXtDiseaseTrainer:
    """
    Trainer for ConvNeXt disease classification model.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = []
        self.best_accuracy = 0.0
        
        logger.info(f"Using device: {self.device}")
    
    @staticmethod
    def _default_config() -> Dict:
        """Default training configuration."""
        return {
            # Model
            'model_name': 'convnextv2_nano',
            'pretrained': True,
            'drop_rate': 0.1,
            
            # Training
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 0.05,
            'warmup_epochs': 5,
            'min_lr': 1e-6,
            
            # Data
            'input_size': 384,
            'progressive_resize': True,
            'resize_schedule': [(224, 30), (320, 60), (384, 100)],
            
            # Augmentation
            'copy_paste_prob': 0.3,
            'mixup_alpha': 0.8,
            'mixup_prob': 0.5,
            'label_smoothing': 0.1,
            
            # Training tricks
            'use_amp': True,  # Automatic Mixed Precision
            'gradient_clip': 1.0,
            'early_stopping': 15,
            
            # Output
            'output_dir': './models/vision',
            'model_name_save': 'convnext_disease_v1',
            'seed': 42
        }
    
    def _get_transforms(self, input_size: int, is_train: bool = True):
        """Get image transforms."""
        if ALBUMENTATIONS_AVAILABLE and is_train:
            return A.Compose([
                A.RandomResizedCrop(input_size, input_size, scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=30, p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=7),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ], p=0.5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        elif TORCHVISION_AVAILABLE:
            if is_train:
                return transforms.Compose([
                    transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(30),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(int(input_size * 1.14)),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            raise ImportError("torchvision or albumentations required")
    
    def _create_dataloaders(self, 
                            data_dir: str,
                            input_size: int,
                            background_dir: str = None) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        data_dir = Path(data_dir)
        
        # Check if we have train/val split
        train_dir = data_dir / 'train'
        val_dir = data_dir / 'val'
        
        if train_dir.exists() and val_dir.exists():
            train_dataset = DiseaseDataset(
                train_dir,
                transform=self._get_transforms(input_size, is_train=True),
                copy_paste_prob=self.config['copy_paste_prob'],
                background_dir=background_dir
            )
            val_dataset = DiseaseDataset(
                val_dir,
                transform=self._get_transforms(input_size, is_train=False)
            )
        else:
            # Create single dataset and split
            full_dataset = DiseaseDataset(
                data_dir,
                transform=None  # Apply transforms after split
            )
            
            # 80-20 split
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            train_indices, val_indices = torch.utils.data.random_split(
                range(len(full_dataset)), [train_size, val_size],
                generator=torch.Generator().manual_seed(self.config['seed'])
            )
            
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
            val_dataset = torch.utils.data.Subset(full_dataset, val_indices.indices)
            
            # Apply transforms
            train_dataset.dataset.transform = self._get_transforms(input_size, is_train=True)
            val_dataset.dataset.transform = self._get_transforms(input_size, is_train=False)
            
            self.class_names = full_dataset.classes
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train(self, data_dir: str, background_dir: str = None) -> Dict:
        """
        Train the ConvNeXt disease classifier.
        
        Args:
            data_dir: Directory with disease images
            background_dir: Directory with field backgrounds
            
        Returns:
            Dict with training metrics
        """
        if not TORCH_AVAILABLE or not TIMM_AVAILABLE:
            raise ImportError("PyTorch and timm are required")
        
        logger.info("=" * 60)
        logger.info("🌿 ConvNeXt V2 Disease Detection Training")
        logger.info("=" * 60)
        
        # Set seed
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        
        # Get initial input size
        if self.config['progressive_resize']:
            input_size, _ = self.config['resize_schedule'][0]
        else:
            input_size = self.config['input_size']
        
        # Create dataloaders
        train_loader, val_loader = self._create_dataloaders(
            data_dir, input_size, background_dir
        )
        
        num_classes = len(self.class_names) if self.class_names else len(train_loader.dataset.dataset.classes)
        if hasattr(train_loader.dataset, 'dataset'):
            self.class_names = train_loader.dataset.dataset.classes
        
        logger.info(f"Classes: {num_classes}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Create model
        self.model = ConvNeXtDiseaseClassifier(
            num_classes=num_classes,
            pretrained=self.config['pretrained'],
            drop_rate=self.config['drop_rate']
        ).to(self.device)
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.config['epochs'],
            warmup_t=self.config['warmup_epochs'],
            warmup_lr_init=self.config['min_lr'],
            lr_min=self.config['min_lr']
        )
        
        # Loss function
        if self.config['mixup_prob'] > 0:
            criterion = SoftTargetCrossEntropy()
            mixup = Mixup(self.config['mixup_alpha'], self.config['mixup_prob'])
        else:
            criterion = LabelSmoothingCrossEntropy(smoothing=self.config['label_smoothing'])
            mixup = None
        
        # Mixed precision
        scaler = GradScaler() if self.config['use_amp'] else None
        
        # Training loop
        metrics_history = []
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Progressive resize
            if self.config['progressive_resize']:
                for size, end_epoch in self.config['resize_schedule']:
                    if epoch < end_epoch:
                        if size != input_size:
                            input_size = size
                            train_loader, val_loader = self._create_dataloaders(
                                data_dir, input_size, background_dir
                            )
                            logger.info(f"Resizing to {input_size}x{input_size}")
                        break
            
            # Train epoch
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Mixup
                if mixup:
                    images, labels_a, labels_b, lam = mixup(images, labels)
                    labels_mixed = lam * torch.nn.functional.one_hot(labels_a, num_classes).float() + \
                                   (1 - lam) * torch.nn.functional.one_hot(labels_b, num_classes).float()
                
                optimizer.zero_grad()
                
                if self.config['use_amp']:
                    with autocast():
                        outputs = self.model(images)
                        if mixup:
                            loss = criterion(outputs, labels_mixed)
                        else:
                            loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    if self.config['gradient_clip']:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['gradient_clip']
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(images)
                    if mixup:
                        loss = criterion(outputs, labels_mixed)
                    else:
                        loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                if not mixup:
                    train_correct += predicted.eq(labels).sum().item()
            
            scheduler.step(epoch)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = 100. * val_correct / val_total
            
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            metrics_history.append(metrics)
            
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']} | "
                       f"Train Loss: {train_loss:.4f} | "
                       f"Val Loss: {val_loss:.4f} | "
                       f"Val Acc: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                best_epoch = epoch + 1
                patience_counter = 0
                self._save_checkpoint('best')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['early_stopping']:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        logger.info(f"\n🏆 Best accuracy: {self.best_accuracy:.2f}% at epoch {best_epoch}")
        
        return {
            'best_accuracy': self.best_accuracy,
            'best_epoch': best_epoch,
            'history': metrics_history
        }
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'class_names': self.class_names,
            'best_accuracy': self.best_accuracy
        }
        
        path = output_dir / f"{self.config['model_name_save']}_{name}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def save_model(self, output_dir: str = None) -> Dict[str, str]:
        """Save model in multiple formats."""
        if self.model is None:
            raise ValueError("No trained model")
        
        output_dir = Path(output_dir or self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        model_name = self.config['model_name_save']
        
        # Save PyTorch model
        torch_path = output_dir / f'{model_name}.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'class_names': self.class_names,
            'best_accuracy': self.best_accuracy
        }, torch_path)
        saved_files['pytorch'] = str(torch_path)
        
        # Save ONNX
        try:
            self.model.eval()
            dummy_input = torch.randn(1, 3, self.config['input_size'], 
                                      self.config['input_size']).to(self.device)
            onnx_path = output_dir / f'{model_name}.onnx'
            
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=14,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            saved_files['onnx'] = str(onnx_path)
            logger.info(f"✓ Saved ONNX: {onnx_path}")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
        
        # Save class names
        classes_path = output_dir / f'{model_name}_classes.json'
        with open(classes_path, 'w') as f:
            json.dump({
                'classes': self.class_names,
                'num_classes': len(self.class_names),
                'input_size': self.config['input_size'],
                'best_accuracy': self.best_accuracy
            }, f, indent=2)
        saved_files['metadata'] = str(classes_path)
        
        return saved_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train ConvNeXt disease classifier')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with disease images')
    parser.add_argument('--background-dir', type=str, default=None,
                        help='Directory with field backgrounds')
    parser.add_argument('--output-dir', type=str, default='./models/vision',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input-size', type=int, default=384)
    parser.add_argument('--no-progressive', action='store_true',
                        help='Disable progressive resizing')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🌿 AgriSense ConvNeXt Disease Trainer")
    print("=" * 70)
    
    # Check data directory
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Configure trainer
    config = ConvNeXtDiseaseTrainer._default_config()
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['input_size'] = args.input_size
    config['output_dir'] = args.output_dir
    config['progressive_resize'] = not args.no_progressive
    
    trainer = ConvNeXtDiseaseTrainer(config)
    metrics = trainer.train(args.data_dir, args.background_dir)
    saved_files = trainer.save_model()
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"Best accuracy: {metrics['best_accuracy']:.2f}%")
    print(f"\nSaved files:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")


if __name__ == '__main__':
    main()
