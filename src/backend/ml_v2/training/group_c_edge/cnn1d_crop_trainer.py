"""
1D-CNN Crop Prediction Model with Quantization Aware Training (QAT)

This module implements a MobileNetV3-inspired 1D-CNN architecture for crop prediction
optimized for ESP32-S3 deployment with TFLite Micro.

Architecture Highlights:
    - 1D Convolutions capture inter-feature relationships (N:P:K ratios)
    - Squeeze-and-Excitation (SE) blocks for channel attention
    - Hard-Swish activation for better gradients
    - Quantization Aware Training for INT8 deployment
    - Target: <500KB model, <50ms inference on ESP32-S3

Usage:
    trainer = CNN1DCropTrainer(config_path="config/training_config.yaml")
    trainer.train()
    trainer.export_tflite("models/group_c_edge/crop_cnn1d_int8.tflite")
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import tensorflow_model_optimization as tfmot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# CUSTOM LAYERS: MobileNetV3-Inspired Blocks for 1D
# =============================================================================

class HardSwish(layers.Layer):
    """
    Hard-Swish activation: x * ReLU6(x + 3) / 6
    More efficient than regular Swish for edge deployment.
    """
    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0


class HardSigmoid(layers.Layer):
    """Hard-Sigmoid: ReLU6(x + 3) / 6"""
    def call(self, x):
        return tf.nn.relu6(x + 3.0) / 6.0


class SqueezeExcitation1D(layers.Layer):
    """
    Squeeze-and-Excitation block for 1D inputs.
    
    Learns channel-wise attention weights to emphasize important features
    (e.g., N:P:K ratio relationships).
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck
    """
    def __init__(self, channels: int, reduction: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        
        reduced_channels = max(1, channels // reduction)
        
        self.global_pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(reduced_channels, activation='relu')
        self.fc2 = layers.Dense(channels)
        self.sigmoid = HardSigmoid()
    
    def call(self, x):
        # Squeeze: Global Average Pooling
        squeeze = self.global_pool(x)  # (batch, channels)
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        excite = self.fc1(squeeze)
        excite = self.fc2(excite)
        excite = self.sigmoid(excite)
        
        # Reshape for broadcasting: (batch, 1, channels)
        excite = tf.expand_dims(excite, axis=1)
        
        # Scale: element-wise multiplication
        return x * excite
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction": self.reduction
        })
        return config


class InvertedResidual1D(layers.Layer):
    """
    MobileNetV3-style Inverted Residual Block for 1D data.
    
    Structure: Expand -> Depthwise Conv -> SE -> Project
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        expansion: Expansion factor for hidden dim
        use_se: Whether to use Squeeze-and-Excitation
        activation: 'relu' or 'hard_swish'
        stride: Convolution stride
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        expansion: int = 4,
        use_se: bool = True,
        activation: str = 'hard_swish',
        stride: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        hidden_dim = int(in_channels * expansion)
        
        # Activation function
        act_fn = HardSwish() if activation == 'hard_swish' else layers.ReLU()
        
        # Expansion layer (1x1 conv)
        self.expand = Sequential([
            layers.Conv1D(hidden_dim, 1, use_bias=False),
            layers.BatchNormalization(),
            act_fn
        ]) if expansion != 1 else None
        
        # Depthwise convolution
        self.depthwise = Sequential([
            layers.Conv1D(
                hidden_dim, kernel_size, 
                strides=stride, 
                padding='same',
                groups=hidden_dim,  # Depthwise
                use_bias=False
            ),
            layers.BatchNormalization(),
            act_fn
        ])
        
        # Squeeze-and-Excitation
        self.se = SqueezeExcitation1D(hidden_dim) if use_se else None
        
        # Projection layer (1x1 conv, no activation)
        self.project = Sequential([
            layers.Conv1D(out_channels, 1, use_bias=False),
            layers.BatchNormalization()
        ])
    
    def call(self, x, training=None):
        residual = x
        
        # Expand
        if self.expand is not None:
            x = self.expand(x, training=training)
        
        # Depthwise
        x = self.depthwise(x, training=training)
        
        # SE
        if self.se is not None:
            x = self.se(x)
        
        # Project
        x = self.project(x, training=training)
        
        # Residual connection
        if self.use_residual:
            x = x + residual
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "stride": self.stride
        })
        return config


# =============================================================================
# MAIN MODEL: 1D-CNN for Crop Prediction
# =============================================================================

def build_crop_cnn1d(
    input_features: int = 19,
    num_classes: int = 96,
    channels: List[int] = [16, 24, 40, 48],
    kernel_sizes: List[int] = [3, 3, 5, 3],
    use_se: bool = True,
    dropout_rate: float = 0.2
) -> Model:
    """
    Build MobileNetV3-inspired 1D-CNN for crop prediction.
    
    Architecture:
        Input (19 features) -> Reshape to (19, 1)
        -> Stem Conv (expand channels)
        -> 4x InvertedResidual1D blocks
        -> Global Average Pooling
        -> Classifier Head
    
    Args:
        input_features: Number of input sensor features
        num_classes: Number of crop classes (96 for India dataset)
        channels: Output channels for each block
        kernel_sizes: Kernel sizes for each block
        use_se: Use Squeeze-and-Excitation
        dropout_rate: Dropout rate before classifier
        
    Returns:
        Keras Model ready for training
    """
    inputs = layers.Input(shape=(input_features,), name='sensor_input')
    
    # Reshape: (batch, features) -> (batch, features, 1)
    # This allows 1D convolutions to process feature relationships
    x = layers.Reshape((input_features, 1))(inputs)
    
    # Stem: Initial convolution to expand channels
    x = layers.Conv1D(channels[0], 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = HardSwish()(x)
    
    # Inverted Residual Blocks
    in_channels = channels[0]
    for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
        x = InvertedResidual1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            expansion=4 if i > 0 else 1,  # No expansion in first block
            use_se=use_se,
            activation='hard_swish',
            stride=1,  # No downsampling (sequence is already short)
            name=f'block_{i}'
        )(x)
        in_channels = out_channels
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classifier head
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = HardSwish()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='crop_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CropCNN1D')
    
    return model


# =============================================================================
# QUANTIZATION AWARE TRAINING (QAT)
# =============================================================================

def apply_quantization_aware_training(model: Model) -> Model:
    """
    Apply Quantization Aware Training to the model.
    
    QAT inserts fake quantization nodes during training to simulate
    INT8 quantization effects, minimizing accuracy loss during conversion.
    
    Args:
        model: Keras model to quantize
        
    Returns:
        QAT-enabled model (same architecture, with quantization wrappers)
    """
    # Define quantization configuration
    # Use default 8-bit quantization for weights and activations
    quantize_model = tfmot.quantization.keras.quantize_model
    
    # Clone and apply QAT
    qat_model = quantize_model(model)
    
    logger.info("Applied Quantization Aware Training to model")
    logger.info(f"  - Original params: {model.count_params():,}")
    logger.info(f"  - QAT params: {qat_model.count_params():,}")
    
    return qat_model


def convert_to_tflite_int8(
    model: Model,
    representative_dataset: tf.data.Dataset,
    output_path: Path
) -> None:
    """
    Convert QAT model to TFLite INT8 format.
    
    Args:
        model: QAT-trained Keras model
        representative_dataset: Dataset for calibration
        output_path: Path to save .tflite file
    """
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Representative dataset for calibration
    def representative_data_gen():
        for batch in representative_dataset.take(100):
            # Only need input tensor
            if isinstance(batch, tuple):
                yield [batch[0].numpy().astype(np.float32)]
            else:
                yield [batch.numpy().astype(np.float32)]
    
    converter.representative_dataset = representative_data_gen
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    
    # Log size
    size_kb = len(tflite_model) / 1024
    logger.info(f"Saved TFLite INT8 model: {output_path}")
    logger.info(f"  - Size: {size_kb:.1f} KB")
    
    return size_kb


# =============================================================================
# TRAINER CLASS
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    input_features: int = 19
    num_classes: int = 96
    channels: List[int] = None
    kernel_sizes: List[int] = None
    use_se: bool = True
    
    # Training params
    batch_size: int = 64
    epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    # QAT params
    qat_enabled: bool = True
    qat_start_epoch: int = 50
    
    # Paths
    data_path: str = None
    checkpoint_dir: str = "checkpoints/crop_cnn1d"
    log_dir: str = "logs/crop_cnn1d"
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [16, 24, 40, 48]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 5, 3]


class CNN1DCropTrainer:
    """
    Complete training pipeline for 1D-CNN Crop Prediction with QAT.
    
    Usage:
        trainer = CNN1DCropTrainer()
        trainer.load_data("data/processed/crop_recommendation")
        trainer.train()
        trainer.export_tflite("models/crop_cnn1d_int8.tflite")
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize trainer with configuration."""
        self.config = config or TrainingConfig()
        self.model = None
        self.qat_model = None
        self.history = None
        
        # Data
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.label_encoder = None
        
        # Create directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def load_data(
        self, 
        data_path: Optional[str] = None,
        csv_path: Optional[str] = None
    ) -> None:
        """
        Load and preprocess training data.
        
        Args:
            data_path: Path to preprocessed pickle file
            csv_path: Path to raw CSV file (alternative)
        """
        if csv_path:
            self._load_from_csv(csv_path)
        elif data_path:
            self._load_from_pickle(data_path)
        else:
            # Generate synthetic data for testing
            logger.warning("No data path provided. Using synthetic data.")
            self._generate_synthetic_data()
    
    def _load_from_csv(self, csv_path: str) -> None:
        """Load data from CSV file."""
        df = pd.read_csv(csv_path)
        
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Update num_classes
        self.config.num_classes = len(self.label_encoder.classes_)
        self.config.input_features = X.shape[1]
        
        # Split
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
        )
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        
        logger.info(f"Loaded data from CSV: {csv_path}")
        logger.info(f"  - Train: {self.X_train.shape}, Val: {self.X_val.shape}")
        logger.info(f"  - Classes: {self.config.num_classes}")
    
    def _load_from_pickle(self, data_path: str) -> None:
        """Load preprocessed data from pickle."""
        import pickle
        
        pkl_path = Path(data_path) / "crop_recommendation_complete.pkl"
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.X_train = data['X_train']
        self.X_val = data['X_test']
        self.y_train = data['y_train']
        self.y_val = data['y_test']
        
        self.config.input_features = self.X_train.shape[1]
        self.config.num_classes = len(np.unique(self.y_train))
        
        logger.info(f"Loaded preprocessed data: {pkl_path}")
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic data for testing."""
        n_samples = 10000
        n_features = self.config.input_features
        n_classes = self.config.num_classes
        
        # Random features
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples)
        
        # Split
        split = int(0.8 * n_samples)
        self.X_train, self.X_val = X[:split], X[split:]
        self.y_train, self.y_val = y[:split], y[split:]
        
        logger.info(f"Generated synthetic data: {n_samples} samples")
    
    def build_model(self) -> Model:
        """Build the 1D-CNN model."""
        self.model = build_crop_cnn1d(
            input_features=self.config.input_features,
            num_classes=self.config.num_classes,
            channels=self.config.channels,
            kernel_sizes=self.config.kernel_sizes,
            use_se=self.config.use_se
        )
        
        logger.info("Built 1D-CNN model:")
        self.model.summary(print_fn=logger.info)
        
        return self.model
    
    def compile_model(self, model: Model) -> None:
        """Compile model with optimizer and loss."""
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_callbacks(self, phase: str = "pretrain") -> List:
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f"{self.config.checkpoint_dir}/{phase}_best.keras",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=f"{self.config.log_dir}/{phase}",
                histogram_freq=1
            )
        ]
        return callbacks
    
    def train(self) -> Dict[str, Any]:
        """
        Complete training pipeline with optional QAT.
        
        Training phases:
            1. Pre-training (epochs 0 to qat_start_epoch): Regular training
            2. QAT Fine-tuning (qat_start_epoch to end): With fake quantization
        
        Returns:
            Training history and metrics
        """
        if self.X_train is None:
            raise ValueError("No training data. Call load_data() first.")
        
        # Build model
        self.build_model()
        self.compile_model(self.model)
        
        # Create tf.data.Dataset for efficient training
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_train.astype(np.float32), self.y_train)
        ).shuffle(10000).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_val.astype(np.float32), self.y_val)
        ).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        results = {}
        
        # Phase 1: Pre-training (without QAT)
        if self.config.qat_enabled and self.config.qat_start_epoch > 0:
            logger.info("=" * 60)
            logger.info("Phase 1: Pre-training (without QAT)")
            logger.info("=" * 60)
            
            history_pretrain = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config.qat_start_epoch,
                callbacks=self.get_callbacks("pretrain"),
                verbose=1
            )
            
            results['pretrain_history'] = history_pretrain.history
            
            # Evaluate
            pretrain_metrics = self.model.evaluate(val_dataset, verbose=0)
            logger.info(f"Pre-training complete. Val Accuracy: {pretrain_metrics[1]:.4f}")
        
        # Phase 2: QAT Fine-tuning
        if self.config.qat_enabled:
            logger.info("=" * 60)
            logger.info("Phase 2: Quantization Aware Training (QAT)")
            logger.info("=" * 60)
            
            # Apply QAT
            self.qat_model = apply_quantization_aware_training(self.model)
            
            # Recompile with lower learning rate
            qat_optimizer = keras.optimizers.AdamW(
                learning_rate=self.config.learning_rate * 0.1,  # Lower LR for QAT
                weight_decay=self.config.weight_decay
            )
            self.qat_model.compile(
                optimizer=qat_optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # QAT training
            remaining_epochs = self.config.epochs - self.config.qat_start_epoch
            history_qat = self.qat_model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=remaining_epochs,
                callbacks=self.get_callbacks("qat"),
                verbose=1
            )
            
            results['qat_history'] = history_qat.history
            
            # Final evaluation
            final_metrics = self.qat_model.evaluate(val_dataset, verbose=0)
            logger.info(f"QAT training complete. Final Val Accuracy: {final_metrics[1]:.4f}")
            
            results['final_accuracy'] = final_metrics[1]
            results['final_loss'] = final_metrics[0]
        else:
            # Train without QAT
            logger.info("Training without QAT")
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config.epochs,
                callbacks=self.get_callbacks("full"),
                verbose=1
            )
            results['history'] = history.history
            
            final_metrics = self.model.evaluate(val_dataset, verbose=0)
            results['final_accuracy'] = final_metrics[1]
            results['final_loss'] = final_metrics[0]
        
        self.history = results
        return results
    
    def export_tflite(
        self, 
        output_path: str = "models/group_c_edge/crop_cnn1d_int8.tflite"
    ) -> float:
        """
        Export trained model to TFLite INT8 format.
        
        Args:
            output_path: Path to save .tflite file
            
        Returns:
            Model size in KB
        """
        # Use QAT model if available, otherwise regular model
        model_to_export = self.qat_model if self.qat_model else self.model
        
        if model_to_export is None:
            raise ValueError("No trained model. Call train() first.")
        
        # Create representative dataset
        rep_dataset = tf.data.Dataset.from_tensor_slices(
            self.X_train[:500].astype(np.float32)
        ).batch(1)
        
        # Convert and save
        size_kb = convert_to_tflite_int8(
            model_to_export,
            rep_dataset,
            Path(output_path)
        )
        
        # Also save metadata
        metadata = {
            "model_name": "CropCNN1D",
            "version": "2.0.0",
            "input_features": self.config.input_features,
            "num_classes": self.config.num_classes,
            "quantization": "INT8",
            "target_device": "ESP32-S3",
            "size_kb": size_kb,
            "accuracy": self.history.get('final_accuracy') if self.history else None
        }
        
        metadata_path = Path(output_path).parent / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
        
        return size_kb
    
    def generate_esp32_header(
        self,
        tflite_path: str,
        output_header: str = "edge/tflite_micro/crop_model_data.h"
    ) -> None:
        """
        Generate C header file for ESP32 deployment.
        
        Converts .tflite binary to C array for embedding in firmware.
        
        Args:
            tflite_path: Path to .tflite model
            output_header: Path to output .h file
        """
        # Read TFLite model
        with open(tflite_path, 'rb') as f:
            model_data = f.read()
        
        # Convert to C array
        hex_array = ', '.join(f'0x{b:02x}' for b in model_data)
        
        header_content = f'''// Auto-generated by CNN1DCropTrainer
// Model: CropCNN1D INT8
// Target: ESP32-S3 with TFLite Micro

#ifndef CROP_MODEL_DATA_H
#define CROP_MODEL_DATA_H

#include <stdint.h>

// Model size: {len(model_data)} bytes ({len(model_data)/1024:.1f} KB)
alignas(8) const uint8_t crop_model_data[] = {{
    {hex_array}
}};

const unsigned int crop_model_data_len = {len(model_data)};

// Model info
#define CROP_MODEL_INPUT_FEATURES {self.config.input_features}
#define CROP_MODEL_NUM_CLASSES {self.config.num_classes}
#define CROP_MODEL_ARENA_SIZE 16384  // Adjust based on profiling

#endif // CROP_MODEL_DATA_H
'''
        
        output_path = Path(output_header)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(header_content)
        
        logger.info(f"Generated ESP32 header: {output_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train 1D-CNN Crop Model with QAT")
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument("--csv", type=str, help="Path to CSV file")
    parser.add_argument("--epochs", type=int, default=200, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--qat-start", type=int, default=50, help="Epoch to start QAT")
    parser.add_argument("--no-qat", action="store_true", help="Disable QAT")
    parser.add_argument("--output", type=str, default="models/group_c_edge/crop_cnn1d_int8.tflite")
    args = parser.parse_args()
    
    # Configuration
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        qat_enabled=not args.no_qat,
        qat_start_epoch=args.qat_start
    )
    
    # Initialize trainer
    trainer = CNN1DCropTrainer(config)
    
    # Load data
    if args.csv:
        trainer.load_data(csv_path=args.csv)
    elif args.data:
        trainer.load_data(data_path=args.data)
    else:
        # Use synthetic data for demonstration
        trainer.load_data()
    
    # Train
    results = trainer.train()
    
    # Export
    trainer.export_tflite(args.output)
    
    # Generate ESP32 header
    trainer.generate_esp32_header(
        args.output,
        "edge/tflite_micro/crop_model_data.h"
    )
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Final Accuracy: {results.get('final_accuracy', 'N/A'):.4f}")
    logger.info(f"  Model saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
