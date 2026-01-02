#!/bin/bash
# GPU-Accelerated Model Retraining for AgriSense
# Activates TensorFlow GPU environment and runs all training pipelines

set -e

echo "🚀 AgriSense GPU Model Retraining Pipeline"
echo "==========================================="
echo ""

# Activate TensorFlow GPU environment
echo "📦 Activating TensorFlow GPU environment..."
source ~/tf_gpu_env/bin/activate

# Navigate to project
cd /mnt/d/AGRISENSEFULL-STACK

echo "✅ Environment activated"
echo ""

# Verify GPU
echo "🔍 Verifying GPU access..."
python -c "import tensorflow as tf; print(f'GPU Devices: {len(tf.config.list_physical_devices(\"GPU\"))}'); print(f'TensorFlow: {tf.__version__}')"

echo ""
echo "🎯 Starting model retraining..."
echo "==============================="
echo ""

# Run retraining orchestrator
python retrain_all_models_gpu.py \
    --backend-path agrisense_app/backend \
    "$@"

echo ""
echo "✅ Retraining pipeline complete!"
echo "Check the models/ directory for updated model files."
