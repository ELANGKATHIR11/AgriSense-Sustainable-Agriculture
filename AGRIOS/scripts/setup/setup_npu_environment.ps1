# ===================================================================
# Intel Core Ultra 9 275HX NPU Environment Setup
# AgriSense ML Model Training Optimization
# ===================================================================

Write-Host "🚀 Setting up Intel NPU environment for AgriSense..." -ForegroundColor Cyan
Write-Host "Target Hardware: Intel Core Ultra 9 275HX with NPU" -ForegroundColor Green

# Check Python version
Write-Host "`n📌 Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Create NPU virtual environment
Write-Host "`n📦 Creating NPU-optimized virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv_npu") {
    Write-Host "Removing existing venv_npu..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv_npu
}
python -m venv venv_npu
Write-Host "✅ Virtual environment created" -ForegroundColor Green

# Activate environment
Write-Host "`n🔌 Activating NPU environment..." -ForegroundColor Yellow
.\venv_npu\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`n⬆️ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install base requirements
Write-Host "`n📥 Installing base ML requirements..." -ForegroundColor Yellow
pip install -r agrisense_app/backend/requirements.txt

# Install NPU optimization stack
Write-Host "`n🧠 Installing Intel NPU optimization tools..." -ForegroundColor Yellow
pip install -r requirements-npu.txt

# Install additional development tools
Write-Host "`n🛠️ Installing development utilities..." -ForegroundColor Yellow
pip install jupyter ipykernel tensorboard

# Configure Intel OpenVINO
Write-Host "`n⚙️ Configuring Intel OpenVINO..." -ForegroundColor Yellow
# Check if OpenVINO installed correctly
python -c "import openvino; print(f'OpenVINO version: {openvino.__version__}')"

# Enable Intel Extensions
Write-Host "`n⚡ Enabling Intel PyTorch Extensions..." -ForegroundColor Yellow
python -c "import intel_extension_for_pytorch as ipex; print(f'IPEX version: {ipex.__version__}')"

# Check NPU availability
Write-Host "`n🔍 Checking NPU device availability..." -ForegroundColor Yellow
python tools/npu/check_npu_devices.py

Write-Host "`n✅ NPU environment setup complete!" -ForegroundColor Green
Write-Host "
📋 Next steps:
1. Run: python tools/npu/benchmark_hardware.py
2. Train models: python tools/npu/train_npu_optimized.py
3. Monitor performance: python tools/npu/monitor_training.py
" -ForegroundColor Cyan
