<#
Setup the repository for development on Windows.
Creates a Python virtual environment, installs dependencies, and initializes Git LFS.
Usage: Open PowerShell in repo root and run: `.	ools\setup_repo.ps1` or `.
ecipes\setup_repo.ps1` depending on location.
#>
param(
    [string]$VenvName = ".venv",
    [string]$Requirements = "requirements.txt"
)

Write-Host "Setting up repository..."

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found in PATH. Install Python 3.12+ and retry."
    exit 1
}

$venvPath = Join-Path $PWD $VenvName
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
    Write-Host "Created virtual environment: $venvPath"
} else {
    Write-Host "Virtual environment already exists: $venvPath"
}

Write-Host "Activating venv and installing dependencies..."
& "$venvPath\Scripts\Activate.ps1"
if (Test-Path $Requirements) {
    pip install --upgrade pip
    pip install -r $Requirements
} else {
    Write-Host "No requirements.txt found at $Requirements. Skipping pip install."
}

Write-Host "Initializing Git LFS (if available)..."
if (Get-Command git-lfs -ErrorAction SilentlyContinue) {
    git lfs install --skip-repo || Write-Host "git lfs install failed or already set"
    Write-Host "Git LFS initialized"
} else {
    Write-Host "Git LFS not installed. Install from https://git-lfs.github.com if you plan to track large files."
}

Write-Host "Setup complete. Activate the venv with: & $venvPath\Scripts\Activate.ps1"
