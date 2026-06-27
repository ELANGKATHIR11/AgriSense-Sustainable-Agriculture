# CUDA 12.8 Installation - Quick Start

## Problem You Hit
You tried to run PowerShell command in **Command Prompt**. They're different!

---

## Solution: Use the Batch Script

### Step 1: Open Command Prompt as Administrator
1. Press `Win + X`
2. Click **"Command Prompt (Admin)"** or **"Windows Terminal (Admin)"`
3. Navigate to the project:
   ```cmd
   cd D:\AGRISENSEFULL-STACK
   ```

### Step 2: Run the Batch Script
```cmd
install_cuda_wsl2.bat
```

This will:
- ✓ Check WSL2 Ubuntu 24.04 exists
- ✓ Update system packages
- ✓ Download and install CUDA 12.8 (~500MB, takes 5-10 min)
- ✓ Configure environment variables
- ✓ Verify installation

---

## Alternative: Simpler Manual Method

If the batch script fails on CUDA download, run **inside WSL2** instead:

### Step 1: Open WSL2
```cmd
wsl -d Ubuntu-24.04
```

### Step 2: Run the Bash Script
```bash
cd /mnt/d/AGRISENSEFULL-STACK
bash install_cuda_wsl2.sh
```

This is **more reliable** because it runs natively in Linux.

---

## Troubleshooting

### "404 - Page Not Found" when downloading CUDA
- Check internet connection
- NVIDIA servers might be slow/overloaded
- Retry the script

### "CUDA installation failed"
- Ensure at least 10GB free disk space in WSL
- Run: `wsl -d Ubuntu-24.04 -- df -h` to check
- Try manual download: https://developer.nvidia.com/cuda-downloads

### Script says "not recognized as command"
- Make sure you're in **Command Prompt Admin**, not regular cmd
- Or use **Windows PowerShell** (not PowerShell 7)
- Right-click → Run as Administrator

---

## Verify Installation

After script completes, test in WSL:

```bash
wsl -d Ubuntu-24.04
nvcc --version          # Should show CUDA 12.8
nvidia-smi              # Should show your GPU
```

---

## Test TensorFlow GPU

```bash
wsl -d Ubuntu-24.04 << 'EOF'
python3.12 -m venv ~/tf_gpu_env
source ~/tf_gpu_env/bin/activate
pip install tensorflow[and-cuda]==2.20.0
python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
EOF
```

If output says `GPU available: True` ✓, you're done!

---

## Use with AgriSense Backend

Once TensorFlow GPU is installed:

```bash
cd D:\AGRISENSEFULL-STACK\agrisense_app\backend
wsl -d Ubuntu-24.04 -e bash -c "source ~/tf_gpu_env/bin/activate && uvicorn agrisense_app.backend.main:app --reload --host 0.0.0.0 --port 8000"
```

Your backend now uses GPU for ML inference!

---

## Files Created

1. **install_cuda_wsl2.bat** — For Command Prompt (Windows)
2. **install_cuda_wsl2.sh** — For Bash (inside WSL2)
3. **WSL2_CUDA_SETUP_GUIDE.md** — Full detailed manual

Pick any one and run it. **Bash script is most reliable.**
