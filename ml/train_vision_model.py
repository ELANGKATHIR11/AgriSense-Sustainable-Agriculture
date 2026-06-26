"""
AGRISENSE Master Vision Training Pipeline
Triggers training and benchmarking for PyTorch classifiers (EfficientNetV2-S and ConvNeXt-Tiny)
and integrates YOLOv11m and YOLOv11m-Seg metadata registration.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.vision_training.disease_classifier import train_and_benchmark

if __name__ == "__main__":
    print("==================================================")
    print("AGRISENSE VISION ENGINE: MASTER RETRAIN RUN")
    print("==================================================")
    train_and_benchmark()
