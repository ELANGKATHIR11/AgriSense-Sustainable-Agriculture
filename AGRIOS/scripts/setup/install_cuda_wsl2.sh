#!/bin/bash
# WSL2 CUDA 12.8 & cuDNN Installation Script
# Run this inside WSL2 Ubuntu 24.04: bash install_cuda_wsl2.sh

set -e

echo "=== WSL2 CUDA 12.8 & cuDNN Setup ==="
echo ""

# Step 1: Update system
echo "[1/4] Updating system packages..."
sudo apt update -qq
sudo apt upgrade -y -qq
echo "✓ System updated"

# Step 2: Install build tools
echo ""
echo "[2/4] Installing build tools and dependencies..."
sudo apt install -y -qq \
  build-essential libffi-dev libssl-dev python3-dev python3-pip \
  wget curl git ca-certificates
echo "✓ Build tools installed"

# Step 3: Download and install CUDA 12.8
echo ""
echo "[3/4] Installing CUDA Toolkit 12.8..."
cd /tmp
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_linux.run"
CUDA_FILE="cuda_12.8.0_linux.run"

if [ ! -f "$CUDA_FILE" ]; then
    echo "  Downloading CUDA 12.8..."
    wget -q --show-progress "$CUDA_URL"
fi

echo "  Running CUDA installer..."
sudo sh "$CUDA_FILE" --silent --toolkit --no-opengl-libs
rm -f "$CUDA_FILE"
echo "✓ CUDA 12.8 installed to /usr/local/cuda-12.8"

# Step 4: Install cuDNN
echo ""
echo "[4/4] Installing cuDNN 9.1+..."
if sudo apt install -y -qq libcudnn9-dev libcudnn9-runtime 2>/dev/null; then
    echo "✓ cuDNN installed from repositories"
else
    echo "⚠ cuDNN not found in repositories."
    echo "  Manual installation:"
    echo "  1. Visit: https://developer.nvidia.com/cudnn"
    echo "  2. Download cuDNN for Linux (x86_64)"
    echo "  3. Extract and copy to CUDA directory:"
    echo "     tar -xzf cudnn-linux-x86_64-*.tar.xz"
    echo "     sudo cp cudnn-linux-x86_64-*/include/cudnn*.h /usr/local/cuda-12.8/include/"
    echo "     sudo cp cudnn-linux-x86_64-*/lib/libcudnn* /usr/local/cuda-12.8/lib64/"
    echo "     sudo ldconfig"
fi

# Step 5: Configure environment
echo ""
echo "Configuring environment variables..."

# Add to ~/.bashrc if not already present
if ! grep -q "CUDA_HOME=/usr/local/cuda-12.8" ~/.bashrc; then
    cat << 'EOF' | tee -a ~/.bashrc > /dev/null

# CUDA 12.8 Configuration
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    echo "✓ Environment variables added to ~/.bashrc"
fi

# Load the variables for this session
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Step 6: Verification
echo ""
echo "=== Verification ==="
echo ""
echo "CUDA Version:"
nvcc --version || echo "⚠ nvcc not found. CUDA may not be properly installed."
echo ""
echo "CUDA Home: $CUDA_HOME"
echo ""
echo "CUDA Libraries:"
ls -lh $CUDA_HOME/lib64/libcuda.so* 2>/dev/null || echo "⚠ CUDA libraries not found"
echo ""

# Step 7: Install TensorFlow GPU (optional)
read -p "Install TensorFlow 2.20 with GPU support? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating Python virtual environment..."
    python3.12 -m venv ~/tf_gpu_env
    source ~/tf_gpu_env/bin/activate
    
    echo "Installing TensorFlow 2.20..."
    pip install --upgrade pip setuptools wheel -q
    pip install "tensorflow[and-cuda]==2.20.0" -q
    
    echo ""
    echo "=== TensorFlow GPU Test ==="
    python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPU available:', len(gpus) > 0)
if gpus:
    print('GPU devices:', gpus)
    for gpu in gpus:
        print(f'  - {gpu.name}')
else:
    print('No GPU detected. Check CUDA/cuDNN installation.')
"
    echo ""
    echo "✓ TensorFlow GPU environment ready at ~/tf_gpu_env"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate TensorFlow GPU environment in future sessions:"
echo "  source ~/tf_gpu_env/bin/activate"
echo ""
echo "To use with AgriSense backend:"
echo "  cd /mnt/d/AGRISENSEFULL-STACK/agrisense_app/backend"
echo "  source ~/tf_gpu_env/bin/activate"
echo "  uvicorn agrisense_app.backend.main:app --reload --host 0.0.0.0 --port 8000"
