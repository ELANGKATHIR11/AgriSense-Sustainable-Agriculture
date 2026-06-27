#!/bin/bash
# Start AgriSense backend with GPU acceleration in WSL

echo "🚀 Starting AgriSense Backend (GPU Mode)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

source ~/tf_gpu_env/bin/activate

cd /mnt/d/AGRISENSEFULL-STACK

echo "✓ Python Env: $(which python)"
echo "✓ TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "✓ GPU Count: $(python -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices(\"GPU\")))')"
echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

uvicorn agrisense_app.backend.main:app --host 0.0.0.0 --port 8000 --reload
