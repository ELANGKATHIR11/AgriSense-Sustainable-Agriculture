#!/bin/bash
# Production model retraining script

VENV="$HOME/tf_gpu_env"
source $VENV/bin/activate

echo "🚀 Production Model Retraining"
echo "=============================="
echo ""

# Show GPU
echo "📊 GPU Status:"
nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader | head -1
echo ""

cd /mnt/d/AGRISENSEFULL-STACK
python3 retrain_production.py

echo ""
echo "✅ Complete!"
