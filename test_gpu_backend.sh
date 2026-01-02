#!/bin/bash
# Test script to verify backend dependencies in WSL GPU venv

source ~/tf_gpu_env/bin/activate
cd /mnt/d/AGRISENSEFULL-STACK

echo "=== Testing Python Imports ==="
python << 'PYEOF'
try:
    import fastapi
    print("✓ FastAPI available")
except ImportError as e:
    print(f"✗ FastAPI missing: {e}")

try:
    import sqlalchemy
    print("✓ SQLAlchemy available")
except ImportError as e:
    print(f"✗ SQLAlchemy missing: {e}")

try:
    import tensorflow
    print("✓ TensorFlow available")
except ImportError as e:
    print(f"✗ TensorFlow missing: {e}")

try:
    import pydantic
    print("✓ Pydantic available")
except ImportError as e:
    print(f"✗ Pydantic missing: {e}")

PYEOF

echo ""
echo "=== Python Version ==="
python --version

echo ""
echo "=== GPU Check ==="
python -c "import tensorflow as tf; print('GPU Devices:', len(tf.config.list_physical_devices('GPU')))"
