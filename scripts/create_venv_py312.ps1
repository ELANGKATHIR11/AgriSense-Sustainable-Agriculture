<#
PowerShell helper: create_venv_py312.ps1
Creates a Python 3.12 virtual environment named .venv312 and installs optimization requirements.
Usage: Run from repository root in PowerShell (may require running as a normal user, not admin):
    .\scripts\create_venv_py312.ps1
This script checks for `py -3.12`. If not found, it prints instructions to install Python 3.12.
#>

set-strictmode -Version Latest

Write-Host "== AgriSense: Create Python 3.12 virtualenv (.venv312) ==" -ForegroundColor Cyan

# Check for py launcher with 3.12
$py312 = $null
try {
    $py312 = & py -3.12 --version 2>$null
} catch {
    $py312 = $null
}

if (-not $py312) {
    Write-Host "Python 3.12 not found via 'py -3.12'." -ForegroundColor Yellow
    Write-Host "Please install Python 3.12 from https://www.python.org/downloads/ or make it available as 'py -3.12' before running this script." -ForegroundColor Yellow
    exit 2
}

# Create venv
$venvPath = ".venv312"
if (Test-Path $venvPath) {
    Write-Host "Virtualenv '$venvPath' already exists. Skipping creation." -ForegroundColor Green
} else {
    Write-Host "Creating virtualenv using Python 3.12..." -ForegroundColor Cyan
    & py -3.12 -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtualenv." -ForegroundColor Red
        exit 3
    }
    Write-Host "Virtualenv created at $venvPath" -ForegroundColor Green
}

# Activate venv for this script run (this doesn't persist in parent shell)
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "Activation script not found at $activateScript" -ForegroundColor Red
    exit 4
}

Write-Host "Activating virtualenv for this PowerShell session..." -ForegroundColor Cyan
. $activateScript

# Upgrade pip/wheel/setuptools
Write-Host "Upgrading pip, setuptools, wheel..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Write-Host "pip upgrade failed (continuing)..." -ForegroundColor Yellow }

# Install requirements
$reqPath = "agrisense_app/backend/requirements.optimization.txt"
if (-not (Test-Path $reqPath)) {
    Write-Host "Requirements file not found at $reqPath" -ForegroundColor Red
    exit 5
}

Write-Host "Installing optimization requirements from $reqPath..." -ForegroundColor Cyan
try {
    pip install -r $reqPath
} catch {
    Write-Host "pip returned an error. See messages above." -ForegroundColor Yellow
}

# Helpful hint about onnxruntime
Write-Host "\nIf pip fails on 'onnxruntime', try installing the CPU wheel explicitly:" -ForegroundColor Yellow
Write-Host "  pip install onnxruntime==1.16.0" -ForegroundColor Yellow

Write-Host "\nDone. To activate the venv in your interactive shell, run:" -ForegroundColor Green
Write-Host "  .\\.venv312\\Scripts\\Activate.ps1" -ForegroundColor Green
Write-Host "Then you can run: pip install -r agrisense_app/backend/requirements.optimization.txt" -ForegroundColor Green

exit 0
