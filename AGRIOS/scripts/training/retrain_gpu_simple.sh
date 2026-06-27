#!/bin/bash
# Run GPU-accelerated model retraining in WSL

set -e

WORKSPACE="/mnt/d/AGRISENSEFULL-STACK"
VENV="$HOME/tf_gpu_env"

echo "🚀 Starting GPU-Accelerated Model Retraining..."
echo "=================================================="

# Check GPU
echo -e "\n📊 Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Activate venv
echo -e "\n🔧 Activating TensorFlow GPU environment..."
source $VENV/bin/activate

# Check TensorFlow
echo -e "\n🔍 Verifying TensorFlow GPU setup..."
python3 << EOF
import tensorflow as tf
print(f"✅ TensorFlow version: {tf.__version__}")
print(f"✅ GPU Devices: {len(tf.config.list_physical_devices('GPU'))}")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("✅ GPU is available for training!")
else:
    print("⚠️  No GPU detected - training will use CPU")
EOF

# Run retraining
echo -e "\n🎯 Starting model retraining..."
cd $WORKSPACE
python3 retrain_gpu_simple.py

echo -e "\n✅ Retraining complete!"
echo "📁 Check agrisense_app/backend/models/ for trained models"
