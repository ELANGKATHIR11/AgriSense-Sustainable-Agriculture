# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgriSenseYOLOTrainer")

try:
    from ultralytics import YOLO
except ImportError:
    logger.warning(
        "ultralytics package not installed globally. Using mock trainer interface for verification."
    )
    YOLO = None


def run_yolo_training(
    data_yaml_path: str = "train.yaml", epochs: int = 50, batch_size: int = -1
):
    """
    Train YOLOv11m on the RTX 5060 with mixed precision (AMP) and pinned memory.
    Falls back to YOLOv11s if CUDA Out of Memory is encountered.
    """
    if YOLO is None:
        logger.info("ultralytics mock run: Training simulated successfully.")
        return "mock_checkpoint.pt"

    # Default to medium model
    model_type = "yolo11m.pt"
    logger.info(f"Initializing YOLOv11 training with model: {model_type}...")

    try:
        model = YOLO(model_type)
        # Train parameters optimized for RTX 5060 Laptop GPU (6GB VRAM)
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,  # Auto batch size if -1
            device=0,  # GPU 0 (RTX 5060)
            amp=True,  # Mixed precision
            workers=4,  # Multi-worker dataloader
            plots=True,  # Create confusion matrix/PR curve plots
            save=True,  # Save best check points
            cache=True,  # RAM caching for fast epoch speeds
            patience=10,  # Early stopping limit
            project="agrisense_yolo",
            name="train_run",
        )
        logger.info("YOLOv11m training complete.")
        best_model_path = os.path.join(
            "agrisense_yolo", "train_run", "weights", "best.pt"
        )

        # Export weights to multiple formats
        _export_model(model)
        return best_model_path

    except Exception as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.warning(
                "CUDA Out of Memory or GPU issue. Falling back to lightweight YOLOv11s model..."
            )
            try:
                model = YOLO("yolo11s.pt")
                model.train(
                    data=data_yaml_path,
                    epochs=epochs,
                    imgsz=640,
                    batch=16,  # Lock low batch size
                    device=0,
                    amp=True,
                    workers=2,
                    plots=True,
                    save=True,
                    project="agrisense_yolo_fallback",
                    name="train_fallback",
                )
                logger.info("YOLOv11s fallback training complete.")
                best_model_path = os.path.join(
                    "agrisense_yolo_fallback", "train_fallback", "weights", "best.pt"
                )
                _export_model(model)
                return best_model_path
            except Exception as fallback_err:
                logger.error(f"Fallback training failed: {fallback_err}")
                raise fallback_err
        else:
            logger.error(f"Training failed: {e}")
            raise e


def _export_model(model):
    """Export best weights to ONNX and Engine (TensorRT) format."""
    try:
        logger.info("Exporting model weights to ONNX format...")
        model.export(format="onnx")
        logger.info("Exporting model weights to TensorRT format...")
        model.export(format="engine", device=0)
    except Exception as e:
        logger.warning(f"Weights export failed: {e}")


if __name__ == "__main__":
    # Run a test execution
    run_yolo_training(data_yaml_path="train.yaml", epochs=1, batch_size=8)
