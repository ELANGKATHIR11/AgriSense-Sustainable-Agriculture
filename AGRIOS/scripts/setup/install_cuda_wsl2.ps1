# WSL2 CUDA 12.8 & cuDNN Installation Script
# Run this script in Admin PowerShell on Windows

Write-Host "=== WSL2 CUDA 12.8 Setup Script ===" -ForegroundColor Cyan

# Step 1: Check if WSL2 Ubuntu 24.04 exists
Write-Host "`n[1/5] Checking WSL2 distributions..." -ForegroundColor Yellow
$wslList = wsl --list --verbose
if ($wslList -match "Ubuntu-24.04") {
    Write-Host "✓ Ubuntu 24.04 found" -ForegroundColor Green
} else {
    Write-Host "✗ Ubuntu 24.04 not found. Please install it manually:" -ForegroundColor Red
    Write-Host "  wsl --install -d Ubuntu-24.04"
    exit 1
}

# Step 2: Update WSL2 and install build tools
Write-Host "`n[2/5] Updating WSL2 and installing build tools..." -ForegroundColor Yellow
$buildToolsCmd = @"
sudo apt update && sudo apt upgrade -y && \
sudo apt install -y build-essential libffi-dev libssl-dev python3-dev python3-pip \
  wget curl git ca-certificates libnvidia-compute-*
"@
wsl -d Ubuntu-24.04 -- bash -c $buildToolsCmd
Write-Host "✓ Build tools installed" -ForegroundColor Green

# Step 3: Download and install CUDA 12.8
Write-Host "`n[3/5] Installing CUDA Toolkit 12.8..." -ForegroundColor Yellow
$cudaInstallCmd = @"
cd /tmp && \
echo 'Downloading CUDA 12.8...' && \
wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_linux.run && \
echo 'Running CUDA installer...' && \
sudo sh cuda_12.8.0_linux.run --silent --toolkit --no-opengl-libs && \
rm -f cuda_12.8.0_linux.run && \
echo 'CUDA installation complete'
"@
wsl -d Ubuntu-24.04 -- bash -c $cudaInstallCmd
Write-Host "✓ CUDA 12.8 installed" -ForegroundColor Green

# Step 4: Install cuDNN
Write-Host "`n[4/5] Installing cuDNN 9.1+..." -ForegroundColor Yellow
$cudnnCmd = @"
echo 'Installing cuDNN from repos...' && \
sudo apt install -y libcudnn9-dev libcudnn9-runtime || \
echo 'Note: cuDNN packages not in repos. Download manually from https://developer.nvidia.com/cudnn'
"@
wsl -d Ubuntu-24.04 -- bash -c $cudnnCmd
Write-Host "✓ cuDNN installation attempted" -ForegroundColor Green

# Step 5: Configure environment and verify
Write-Host "`n[5/5] Configuring environment and verifying installation..." -ForegroundColor Yellow
$verifyCmd = @"
# Add CUDA to bashrc
echo '' >> ~/.bashrc && \
echo '# CUDA 12.8 Configuration' >> ~/.bashrc && \
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc && \
echo 'export PATH=\$CUDA_HOME/bin:\$PATH' >> ~/.bashrc && \
echo 'export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc && \
source ~/.bashrc && \
echo '=== Verification ===' && \
echo 'CUDA Version:' && \
nvcc --version && \
echo 'CUDA Home:' && \
echo \$CUDA_HOME && \
echo 'CUDA Libraries:' && \
ls -la \$CUDA_HOME/lib64/ | grep libcuda
"@
wsl -d Ubuntu-24.04 -- bash -c $verifyCmd

Write-Host "`n[COMPLETE] CUDA 12.8 setup finished!" -ForegroundColor Green
Write-Host @"

Next steps:
1. In WSL2 (Ubuntu 24.04), create a Python venv and test TensorFlow GPU:
   python3.12 -m venv ~/tf_gpu_env
   source ~/tf_gpu_env/bin/activate
   pip install --upgrade pip
   pip install tensorflow[and-cuda]==2.20.0
   python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"

2. To use GPU in your AgriSense project:
   cd /mnt/d/AGRISENSEFULL-STACK/agrisense_app/backend
   source ~/tf_gpu_env/bin/activate
   pip install -r requirements.txt
   uvicorn agrisense_app.backend.main:app --reload --host 0.0.0.0 --port 8000

For detailed instructions, see: WSL2_CUDA_SETUP_GUIDE.md
"@ -ForegroundColor Cyan
