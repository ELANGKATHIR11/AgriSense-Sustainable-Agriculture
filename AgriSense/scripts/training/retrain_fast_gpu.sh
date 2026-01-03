#!/bin/bash
# Fast GPU-accelerated model retraining

VENV="$HOME/tf_gpu_env"
source $VENV/bin/activate

echo "🚀 Starting Fast GPU Model Retraining..."
echo "========================================"
nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

cd /mnt/d/AGRISENSEFULL-STACK
python3 retrain_fast_gpu.py

echo ""
echo "✅ Retraining Complete!"
