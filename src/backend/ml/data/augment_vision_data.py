#!/usr/bin/env python3
"""
AgriSense Vision Data Augmentation Pipeline
============================================
Copy-Paste augmentation for disease detection and weed segmentation models.

Features:
- Lab background removal (PlantVillage dataset)
- Copy-Paste augmentation with field backgrounds
- Gaussian blending for realistic composites
- Multi-scale leaf insertion
- YOLO/COCO format annotation generation

Usage:
    python augment_vision_data.py --input-dir ./images/diseases --output-dir ./images/augmented

Author: AgriSense ML Team
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
import logging
import argparse
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    min_scale: float = 0.3
    max_scale: float = 1.2
    min_rotation: int = -45
    max_rotation: int = 45
    min_objects_per_image: int = 1
    max_objects_per_image: int = 5
    blur_kernel_range: Tuple[int, int] = (3, 7)
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    gaussian_blur_sigma: float = 2.0
    output_size: Tuple[int, int] = (640, 640)  # YOLO default
    jpeg_quality: int = 95


class LabBackgroundRemover:
    """
    Remove laboratory backgrounds from plant leaf images.
    
    Handles common backgrounds in PlantVillage dataset:
    - Black backgrounds
    - Gray backgrounds
    - Green backgrounds (leaves on tables)
    """
    
    def __init__(self, 
                 black_threshold: int = 30,
                 gray_threshold: int = 50,
                 white_threshold: int = 240):
        self.black_threshold = black_threshold
        self.gray_threshold = gray_threshold
        self.white_threshold = white_threshold
    
    def remove_background(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove lab background from leaf image.
        
        Args:
            image: BGR input image
            
        Returns:
            Tuple of (BGRA image with transparent background, binary mask)
        """
        if image is None:
            raise ValueError("Input image is None")
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Create masks for different background types
        
        # 1. Black background mask
        black_mask = gray < self.black_threshold
        
        # 2. Gray background mask (low saturation, mid-range value)
        gray_bg_mask = (hsv[:, :, 1] < self.gray_threshold) & \
                       (gray > self.black_threshold) & \
                       (gray < self.white_threshold)
        
        # 3. White/bright background mask
        white_mask = gray > self.white_threshold
        
        # Combine background masks
        background_mask = black_mask | gray_bg_mask | white_mask
        
        # Invert to get foreground mask
        foreground_mask = ~background_mask
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(
            foreground_mask.astype(np.uint8) * 255, 
            cv2.MORPH_CLOSE, 
            kernel, 
            iterations=2
        )
        foreground_mask = cv2.morphologyEx(
            foreground_mask, 
            cv2.MORPH_OPEN, 
            kernel, 
            iterations=1
        )
        
        # Find largest contour (main leaf)
        contours, _ = cv2.findContours(
            foreground_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            refined_mask = np.zeros_like(foreground_mask)
            cv2.drawContours(refined_mask, [largest_contour], -1, 255, -1)
            
            # Smooth the mask edges
            refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
        else:
            refined_mask = foreground_mask
        
        # Create BGRA output with alpha channel
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = refined_mask
        
        return bgra, refined_mask
    
    def extract_leaf_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract the main leaf contour from mask."""
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
            
        return max(contours, key=cv2.contourArea)


class CopyPasteAugmentor:
    """
    Copy-Paste augmentation for realistic field composite images.
    
    Based on the paper "Simple Copy-Paste is a Strong Data Augmentation Method"
    with modifications for agricultural imagery.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        self.background_remover = LabBackgroundRemover()
    
    def load_foreground(self, path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load foreground image and remove background.
        
        Args:
            path: Path to foreground image
            
        Returns:
            Tuple of (BGRA image, mask)
        """
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not load image: {path}")
        return self.background_remover.remove_background(image)
    
    def load_background(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load and resize background image.
        
        Args:
            path: Path to background image
            
        Returns:
            BGR image resized to output size
        """
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not load background: {path}")
        return cv2.resize(image, self.config.output_size)
    
    def random_transform(self, 
                         foreground: np.ndarray, 
                         mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random geometric and color transformations.
        
        Args:
            foreground: BGRA foreground image
            mask: Binary mask
            
        Returns:
            Tuple of transformed (foreground, mask)
        """
        h, w = foreground.shape[:2]
        
        # Random scale
        scale = random.uniform(self.config.min_scale, self.config.max_scale)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Random rotation
        angle = random.randint(self.config.min_rotation, self.config.max_rotation)
        
        # Create transformation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Apply transformation
        transformed_fg = cv2.warpAffine(
            foreground, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        transformed_mask = cv2.warpAffine(
            mask, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Random horizontal flip
        if random.random() > 0.5:
            transformed_fg = cv2.flip(transformed_fg, 1)
            transformed_mask = cv2.flip(transformed_mask, 1)
        
        # Random brightness/contrast adjustment
        brightness = random.uniform(*self.config.brightness_range)
        contrast = random.uniform(*self.config.contrast_range)
        
        transformed_fg[:, :, :3] = np.clip(
            contrast * transformed_fg[:, :, :3].astype(np.float32) + (brightness - 1) * 128,
            0, 255
        ).astype(np.uint8)
        
        return transformed_fg, transformed_mask
    
    def gaussian_blend(self,
                       background: np.ndarray,
                       foreground: np.ndarray,
                       mask: np.ndarray,
                       position: Tuple[int, int]) -> Tuple[np.ndarray, Dict]:
        """
        Blend foreground onto background with Gaussian edge smoothing.
        
        Args:
            background: BGR background image
            foreground: BGRA foreground image
            mask: Binary mask
            position: (x, y) position for placement
            
        Returns:
            Tuple of (blended image, bounding box dict)
        """
        bg_h, bg_w = background.shape[:2]
        fg_h, fg_w = foreground.shape[:2]
        x, y = position
        
        # Ensure foreground fits within background
        if x + fg_w > bg_w:
            fg_w = bg_w - x
            foreground = foreground[:, :fg_w]
            mask = mask[:, :fg_w]
        if y + fg_h > bg_h:
            fg_h = bg_h - y
            foreground = foreground[:fg_h, :]
            mask = mask[:fg_h, :]
        if x < 0:
            foreground = foreground[:, -x:]
            mask = mask[:, -x:]
            fg_w = foreground.shape[1]
            x = 0
        if y < 0:
            foreground = foreground[-y:, :]
            mask = mask[-y:, :]
            fg_h = foreground.shape[0]
            y = 0
        
        if fg_w <= 0 or fg_h <= 0:
            return background, None
        
        # Create soft mask with Gaussian blur
        soft_mask = cv2.GaussianBlur(
            mask.astype(np.float32), 
            (0, 0), 
            self.config.gaussian_blur_sigma
        ) / 255.0
        soft_mask = np.expand_dims(soft_mask, axis=2)
        
        # Extract ROI
        roi = background[y:y+fg_h, x:x+fg_w].astype(np.float32)
        fg_rgb = foreground[:, :, :3].astype(np.float32)
        
        # Blend
        blended_roi = (fg_rgb * soft_mask + roi * (1 - soft_mask)).astype(np.uint8)
        result = background.copy()
        result[y:y+fg_h, x:x+fg_w] = blended_roi
        
        # Calculate bounding box (YOLO format: center_x, center_y, width, height - normalized)
        # Find actual object bounds from mask
        non_zero = cv2.findNonZero(mask)
        if non_zero is not None:
            x_box, y_box, w_box, h_box = cv2.boundingRect(non_zero)
            bbox = {
                'x_center': ((x + x_box + w_box/2) / bg_w),
                'y_center': ((y + y_box + h_box/2) / bg_h),
                'width': (w_box / bg_w),
                'height': (h_box / bg_h),
                'x_min': x + x_box,
                'y_min': y + y_box,
                'x_max': x + x_box + w_box,
                'y_max': y + y_box + h_box
            }
        else:
            bbox = None
        
        return result, bbox
    
    def copy_paste_augmentation(self,
                                 foreground_path: Union[str, Path],
                                 background_path: Union[str, Path],
                                 class_id: int = 0) -> Tuple[np.ndarray, List[Dict]]:
        """
        Main augmentation function: paste foreground onto background.
        
        Args:
            foreground_path: Path to leaf/disease image
            background_path: Path to field background image
            class_id: Class label for annotation
            
        Returns:
            Tuple of (augmented image, list of annotations)
        """
        # Load images
        foreground, mask = self.load_foreground(foreground_path)
        background = self.load_background(background_path)
        
        annotations = []
        bg_h, bg_w = background.shape[:2]
        
        # Determine number of objects to paste
        num_objects = random.randint(
            self.config.min_objects_per_image,
            self.config.max_objects_per_image
        )
        
        for _ in range(num_objects):
            # Transform foreground
            transformed_fg, transformed_mask = self.random_transform(foreground.copy(), mask.copy())
            
            # Random position
            fg_h, fg_w = transformed_fg.shape[:2]
            max_x = max(0, bg_w - fg_w // 2)
            max_y = max(0, bg_h - fg_h // 2)
            x = random.randint(-fg_w // 4, max_x)
            y = random.randint(-fg_h // 4, max_y)
            
            # Blend onto background
            background, bbox = self.gaussian_blend(
                background, transformed_fg, transformed_mask, (x, y)
            )
            
            if bbox:
                bbox['class_id'] = class_id
                annotations.append(bbox)
        
        return background, annotations
    
    def process_directory(self,
                          foreground_dir: Path,
                          background_dir: Path,
                          output_dir: Path,
                          augmentations_per_image: int = 5,
                          class_mapping: Dict[str, int] = None) -> Dict:
        """
        Process entire directory of foreground images.
        
        Args:
            foreground_dir: Directory containing disease/weed images
            background_dir: Directory containing field backgrounds
            output_dir: Output directory for augmented images
            augmentations_per_image: Number of augmented versions per input
            class_mapping: Dict mapping subdirectory names to class IDs
            
        Returns:
            Dict with processing statistics
        """
        output_dir = Path(output_dir)
        output_images_dir = output_dir / 'images'
        output_labels_dir = output_dir / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all foreground and background images
        fg_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        foreground_images = []
        
        foreground_dir = Path(foreground_dir)
        for ext in fg_extensions:
            foreground_images.extend(foreground_dir.rglob(f'*{ext}'))
        
        background_dir = Path(background_dir)
        background_images = []
        for ext in fg_extensions:
            background_images.extend(background_dir.rglob(f'*{ext}'))
        
        if not foreground_images:
            logger.warning(f"No foreground images found in {foreground_dir}")
            return {'processed': 0, 'failed': 0}
        
        if not background_images:
            logger.warning(f"No background images found in {background_dir}")
            return {'processed': 0, 'failed': 0}
        
        logger.info(f"Found {len(foreground_images)} foreground images")
        logger.info(f"Found {len(background_images)} background images")
        
        # Auto-generate class mapping if not provided
        if class_mapping is None:
            class_mapping = {}
            subdirs = set(p.parent.name for p in foreground_images)
            for i, subdir in enumerate(sorted(subdirs)):
                class_mapping[subdir] = i
        
        stats = {'processed': 0, 'failed': 0, 'total_augmented': 0}
        
        # Process images
        for fg_path in tqdm(foreground_images, desc="Augmenting images"):
            # Determine class from parent directory
            class_name = fg_path.parent.name
            class_id = class_mapping.get(class_name, 0)
            
            for aug_idx in range(augmentations_per_image):
                try:
                    # Random background
                    bg_path = random.choice(background_images)
                    
                    # Generate augmented image
                    augmented, annotations = self.copy_paste_augmentation(
                        fg_path, bg_path, class_id
                    )
                    
                    # Save image
                    output_name = f"{fg_path.stem}_aug{aug_idx:03d}"
                    cv2.imwrite(
                        str(output_images_dir / f"{output_name}.jpg"),
                        augmented,
                        [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                    )
                    
                    # Save YOLO format labels
                    with open(output_labels_dir / f"{output_name}.txt", 'w') as f:
                        for ann in annotations:
                            f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
                    
                    stats['total_augmented'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {fg_path}: {e}")
                    stats['failed'] += 1
            
            stats['processed'] += 1
        
        # Save class mapping
        with open(output_dir / 'classes.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        # Create dataset.yaml for YOLO training
        dataset_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'images',
            'val': 'images',  # Should be split separately
            'names': {v: k for k, v in class_mapping.items()}
        }
        with open(output_dir / 'dataset.yaml', 'w') as f:
            import yaml
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        logger.info(f"Processing complete: {stats}")
        return stats


def create_sample_backgrounds(output_dir: Path, count: int = 10):
    """
    Generate synthetic field backgrounds for testing.
    
    Creates green/brown gradient images simulating field conditions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(count):
        # Random field-like colors
        base_color = random.choice([
            (40, 80, 40),   # Green grass
            (60, 100, 50),  # Light green
            (30, 60, 80),   # Brown soil
            (50, 80, 60),   # Mixed
        ])
        
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Create gradient
        for y in range(640):
            variation = int(20 * np.sin(y / 50))
            for x in range(640):
                noise = random.randint(-15, 15)
                img[y, x] = [
                    max(0, min(255, base_color[0] + variation + noise)),
                    max(0, min(255, base_color[1] + variation + noise)),
                    max(0, min(255, base_color[2] + noise))
                ]
        
        # Add some texture
        img = cv2.GaussianBlur(img, (5, 5), 0)
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(str(output_dir / f'field_bg_{i:03d}.jpg'), img)
    
    logger.info(f"Created {count} synthetic backgrounds in {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AgriSense Vision Data Augmentation')
    parser.add_argument('--foreground-dir', type=str, 
                        help='Directory with foreground images (diseases/weeds)')
    parser.add_argument('--background-dir', type=str,
                        help='Directory with field background images')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for augmented images')
    parser.add_argument('--augmentations', type=int, default=5,
                        help='Number of augmentations per image (default: 5)')
    parser.add_argument('--create-sample-backgrounds', action='store_true',
                        help='Generate sample field backgrounds for testing')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with single image pair')
    args = parser.parse_args()
    
    # Get default paths
    base_dir = Path(__file__).parent
    
    print("=" * 70)
    print("🌿 AgriSense Vision Augmentation Pipeline")
    print("=" * 70)
    
    if args.create_sample_backgrounds:
        bg_dir = base_dir / 'images' / 'backgrounds'
        create_sample_backgrounds(bg_dir)
        print(f"\n✅ Created sample backgrounds in {bg_dir}")
        return
    
    if args.demo:
        # Demo mode with synthetic data
        print("\n🔬 Running demo augmentation...")
        
        # Create a sample leaf image (green on black)
        demo_dir = base_dir / 'demo'
        demo_dir.mkdir(exist_ok=True)
        
        leaf = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.ellipse(leaf, (100, 100), (60, 80), 0, 0, 360, (50, 150, 50), -1)
        cv2.ellipse(leaf, (100, 100), (50, 70), 0, 0, 360, (70, 180, 70), -1)
        cv2.imwrite(str(demo_dir / 'sample_leaf.jpg'), leaf)
        
        # Create sample background
        create_sample_backgrounds(demo_dir / 'backgrounds', 3)
        
        # Run augmentation
        config = AugmentationConfig(
            min_objects_per_image=2,
            max_objects_per_image=4,
            output_size=(640, 640)
        )
        augmentor = CopyPasteAugmentor(config)
        
        result, annotations = augmentor.copy_paste_augmentation(
            demo_dir / 'sample_leaf.jpg',
            demo_dir / 'backgrounds' / 'field_bg_000.jpg',
            class_id=0
        )
        
        cv2.imwrite(str(demo_dir / 'augmented_demo.jpg'), result)
        print(f"\n✅ Demo complete! Check {demo_dir}")
        print(f"   Annotations: {annotations}")
        return
    
    # Full processing mode
    fg_dir = Path(args.foreground_dir) if args.foreground_dir else base_dir / 'images' / 'diseases'
    bg_dir = Path(args.background_dir) if args.background_dir else base_dir / 'images' / 'backgrounds'
    out_dir = Path(args.output_dir) if args.output_dir else base_dir / 'images' / 'augmented' / 'diseases'
    
    print(f"\nForeground directory: {fg_dir}")
    print(f"Background directory: {bg_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Augmentations per image: {args.augmentations}")
    
    if not fg_dir.exists():
        logger.error(f"Foreground directory not found: {fg_dir}")
        logger.info("Please download PlantVillage dataset first using download_datasets.sh")
        return
    
    if not bg_dir.exists():
        logger.info(f"Creating sample backgrounds in {bg_dir}")
        create_sample_backgrounds(bg_dir, 20)
    
    # Initialize and run augmentation
    config = AugmentationConfig()
    augmentor = CopyPasteAugmentor(config)
    
    stats = augmentor.process_directory(
        fg_dir, bg_dir, out_dir,
        augmentations_per_image=args.augmentations
    )
    
    print("\n" + "=" * 70)
    print("📊 Augmentation Statistics")
    print("=" * 70)
    print(f"  Images processed: {stats['processed']}")
    print(f"  Total augmented: {stats['total_augmented']}")
    print(f"  Failed: {stats['failed']}")
    print(f"\n✅ Output saved to: {out_dir}")


if __name__ == "__main__":
    main()
