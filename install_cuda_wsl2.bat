@echo off
REM WSL2 CUDA 12.8 & cuDNN Installation Script for Command Prompt
REM Run as Administrator

setlocal enabledelayedexpansion

color 0A
echo.
echo ========================================
echo  WSL2 CUDA 12.8 Setup (Command Prompt)
echo ========================================
echo.

REM Check if running as admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    color 0C
    echo [ERROR] This script must be run as Administrator!
    echo Right-click cmd.exe and select "Run as Administrator"
    pause
    exit /b 1
)

REM Step 1: Check WSL2
echo [1/4] Checking WSL2 Ubuntu 24.04...
wsl --list --verbose | findstr "Ubuntu-24.04" >nul
if errorlevel 1 (
    color 0C
    echo [ERROR] Ubuntu-24.04 not found in WSL2
    echo Please install it first:
    echo   wsl --install -d Ubuntu-24.04
    pause
    exit /b 1
)
color 0A
echo [OK] Ubuntu 24.04 found
echo.

REM Step 2: Update system and install build tools
echo [2/4] Updating system and installing build tools...
color 0E
wsl -d Ubuntu-24.04 -- bash -c "sudo apt update -qq && sudo apt upgrade -y -qq && sudo apt install -y -qq build-essential libffi-dev libssl-dev python3-dev python3-pip wget curl git ca-certificates"
if errorlevel 1 (
    color 0C
    echo [ERROR] Failed to update system
    pause
    exit /b 1
)
color 0A
echo [OK] Build tools installed
echo.

REM Step 3: Install CUDA 12.8
echo [3/4] Installing CUDA Toolkit 12.8...
echo This will take 5-10 minutes...
color 0E
wsl -d Ubuntu-24.04 -- bash -c "cd /tmp && rm -f cuda_12.8.0_linux.run && wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_linux.run && sudo sh cuda_12.8.0_linux.run --silent --toolkit --no-opengl-libs && rm -f cuda_12.8.0_linux.run && echo CUDA installation complete"
if errorlevel 1 (
    color 0C
    echo [ERROR] CUDA installation failed
    echo Check: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)
color 0A
echo [OK] CUDA 12.8 installed
echo.

REM Step 4: Configure environment
echo [4/4] Configuring environment...
color 0E
wsl -d Ubuntu-24.04 -- bash -c "grep -q 'CUDA_HOME=/usr/local/cuda-12.8' ~/.bashrc || (echo '' >> ~/.bashrc && echo '# CUDA 12.8 Configuration' >> ~/.bashrc && echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc && echo 'export PATH=\$CUDA_HOME/bin:\$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc) && source ~/.bashrc && nvcc --version && echo 'CUDA Home: '\$CUDA_HOME"
color 0A
echo [OK] Environment configured
echo.

REM Completion message
color 0B
echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Next Steps:
echo.
echo 1. Install NVIDIA GPU Driver (if not already done):
echo    - Go to: https://www.nvidia.com/Download/driverDetails.aspx
echo    - Select your GPU and Windows version
echo    - Download and install, then restart
echo.
echo 2. Test CUDA in WSL:
echo    wsl -d Ubuntu-24.04
echo    nvcc --version
echo.
echo 3. Install TensorFlow GPU:
echo    wsl -d Ubuntu-24.04
echo    python3.12 -m venv ~/tf_gpu_env
echo    source ~/tf_gpu_env/bin/activate
echo    pip install tensorflow[and-cuda]==2.20.0
echo    python -c "import tensorflow as tf; print('GPU:', len(tf.config.list_physical_devices('GPU')))"
echo.
echo 4. Use with AgriSense:
echo    cd D:\AGRISENSEFULL-STACK\agrisense_app\backend
echo    wsl -d Ubuntu-24.04 -e bash -c "source ~/tf_gpu_env/bin/activate; uvicorn agrisense_app.backend.main:app --reload --host 0.0.0.0 --port 8000"
echo.
pause
