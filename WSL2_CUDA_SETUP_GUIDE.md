# WSL2 CUDA 12.8 & cuDNN Setup Guide

## Step 1: Install NVIDIA Driver on Windows (Host) [MANUAL]

**⚠️ You must run this manually in Admin PowerShell — I cannot automate driver installation.**

### Download and Install the NVIDIA Driver
1. Go to https://www.nvidia.com/Download/driverDetails.aspx
2. Select your GPU model and OS (Windows 11/10 64-bit)
3. Download the driver installer (e.g., `NVIDIA-Driver-xxx.xx-Windows-x86_64.exe`)
4. **Right-click → Run as Administrator**
5. Follow the installer wizard (recommend restart when prompted)
6. After installation, verify in PowerShell:
   ```powershell
   nvidia-smi
   ```
   This should display your GPU info and driver version.

**Minimum driver version for CUDA 12.8/13.0 with WSL2: 527.00 or newer**

---

## Step 2: Update WSL2 and Install System Dependencies

Run these commands in **WSL2 Ubuntu 24.04** terminal:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required build tools and dependencies
sudo apt install -y build-essential libffi-dev libssl-dev python3-dev python3-pip
sudo apt install -y wget curl git ca-certificates
sudo apt install -y nvidia-utils  # NVIDIA utilities for WSL
```

---

## Step 3: Install CUDA Toolkit 12.8

**Note:** If your Ubuntu repos don't have CUDA packages, download directly from NVIDIA.

### Option A: Using NVIDIA's runfile installer (Recommended for WSL2)

```bash
# Download CUDA 12.8 runfile for Linux
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_linux.run

# Run the installer (--silent for non-interactive, --toolkit for CUDA toolkit only)
sudo sh cuda_12.8.0_linux.run --silent --toolkit --no-opengl-libs

# Clean up
rm cuda_12.8.0_linux.run
```

**Expected installation path:** `/usr/local/cuda-12.8`

### Option B: Using package manager (if available)

```bash
# Try to install from repos
sudo apt update
sudo apt install -y cuda-toolkit-12-8 libcuda-12-8

# If above fails, proceed with Option A (runfile)
```

### Install cuDNN 9.1+

```bash
# Option 1: Via package (if available)
sudo apt install -y libcudnn9-dev libcudnn9-runtime

# Option 2: Manual installation
# Download from https://developer.nvidia.com/cudnn
# Extract to CUDA directory (replaces Option 1 if you have .tar.xz):
# tar -xzf cudnn-linux-*.tar.xz
# sudo cp cudnn-linux-*/include/cudnn*.h /usr/local/cuda-12.8/include/
# sudo cp cudnn-linux-*/lib/libcudnn* /usr/local/cuda-12.8/lib64/
# sudo ldconfig
```

---

## Step 4: Configure Environment Variables in WSL

Add these lines to your WSL `~/.bashrc`:

```bash
# Add these to the end of ~/.bashrc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=$CUDA_HOME
```

Apply changes:
```bash
source ~/.bashrc
```

---

## Step 5: Verify CUDA Installation in WSL

Run in **WSL2 Ubuntu 24.04** terminal:

```bash
# Check CUDA version
nvcc --version

# Check CUDA capability
nvidia-smi

# Verify CUDA libraries
ls -la /usr/local/cuda-12.8/lib64/
```

Expected output should show CUDA 12.8 and your GPU.

---

## Step 6: Install cuDNN 9.1+ in WSL (Optional - if not auto-installed)

### Option A: Through apt (Recommended if available)
```bash
sudo apt install -y libcudnn9-dev libcudnn9-runtime
```

### Option B: Manual download (if needed)
1. Go to https://developer.nvidia.com/cudnn
2. Download cuDNN for Linux (ubuntu22.04 compatible with 24.04)
3. Upload to WSL and extract:
   ```bash
   tar -xzvf cudnn-linux-x86_64-*.tar.xz
   sudo cp cudnn-linux-x86_64-*/include/cudnn*.h /usr/local/cuda-12.8/include/
   sudo cp cudnn-linux-x86_64-*/lib/libcudnn* /usr/local/cuda-12.8/lib64/
   sudo ldconfig
   ```

---

## Step 7: Install TensorFlow GPU in WSL Venv

Run in **WSL2 Ubuntu 24.04** terminal (in your project directory):

```bash
# Create a Python venv if not already present
python3.12 -m venv tf_gpu_env

# Activate the venv
source tf_gpu_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install TensorFlow 2.20 (GPU-enabled)
pip install tensorflow[and-cuda]==2.20.0

# Verify TensorFlow GPU access
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU available:', tf.config.list_physical_devices('GPU'))
print('CUDA visible:', len(tf.config.list_physical_devices('GPU')) > 0)
"
```

Expected output:
```
TensorFlow version: 2.20.0
GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
CUDA visible: True
```

---

## Step 8: Run Your AgriSense Backend with GPU

```bash
# In WSL2 Ubuntu 24.04
cd /mnt/d/AGRISENSEFULL-STACK/agrisense_app/backend

# Activate your WSL venv
source ~/tf_gpu_env/bin/activate

# Run the backend
uvicorn agrisense_app.backend.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Troubleshooting

### GPU not detected in TensorFlow
```bash
# Check NVIDIA driver is working
nvidia-smi

# Check CUDA is installed correctly
nvcc --version

# Verify environment variables are set
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
```

### "libcuda.so.1 not found" error
```bash
# Ensure NVIDIA container toolkit is available for WSL
sudo apt install -y nvidia-driver-utils

# Or set the library path explicitly
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### CUDA version mismatch
TensorFlow 2.20 officially supports:
- CUDA 12.2 or later
- cuDNN 9.1 or later

Verify compatibility at: https://www.tensorflow.org/install/source#gpu

---

## Quick Status Check Commands

```bash
# Check if WSL2 is running Ubuntu 24.04
wsl --list --verbose

# Start Ubuntu 24.04 from Windows
wsl -d Ubuntu-24.04

# Check inside WSL if GPU is accessible
nvidia-smi

# Test Python + TensorFlow GPU detection
python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"
```

---

## Notes
- WSL2 with NVIDIA GPU support requires Windows 11 21H2 or Windows 10 21H2+
- Your host NVIDIA driver automatically exposes GPU to WSL2 (no separate WSL driver needed)
- cuDNN is now integrated with CUDA; manual installation is often optional
- Keep TensorFlow, CUDA, and cuDNN versions compatible
