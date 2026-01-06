# AgriSense Dataset Download Helper (Windows PowerShell)
# ========================================================
# Downloads external datasets required for training vision models.
#
# Prerequisites:
#   - Kaggle CLI: pip install kaggle
#   - Kaggle API token: %USERPROFILE%\.kaggle\kaggle.json
#
# Usage:
#   .\download_datasets.ps1 -All
#   .\download_datasets.ps1 -PlantVillage -Backgrounds

param(
    [switch]$All,
    [switch]$PlantVillage,
    [switch]$DeepWeeds,
    [switch]$Backgrounds,
    [switch]$IndianCrops,
    [switch]$Stats,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Colors
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) { Write-Output $args }
    $host.UI.RawUI.ForegroundColor = $fc
}

# Paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir = $ScriptDir
$ImagesDir = Join-Path $DataDir "images"
$DiseasesDir = Join-Path $ImagesDir "diseases"
$WeedsDir = Join-Path $ImagesDir "weeds"
$BackgroundsDir = Join-Path $ImagesDir "backgrounds"
$TabularDir = Join-Path $DataDir "tabular"

# Create directories
New-Item -ItemType Directory -Force -Path $DiseasesDir | Out-Null
New-Item -ItemType Directory -Force -Path $WeedsDir | Out-Null
New-Item -ItemType Directory -Force -Path $BackgroundsDir | Out-Null
New-Item -ItemType Directory -Force -Path $TabularDir | Out-Null

function Show-Header {
    Write-Host "============================================" -ForegroundColor Blue
    Write-Host "🌾 AgriSense Dataset Downloader (Windows)" -ForegroundColor Blue
    Write-Host "============================================" -ForegroundColor Blue
    Write-Host "Data directory: $DataDir"
    Write-Host ""
}

function Test-Dependencies {
    Write-Host "Checking dependencies..." -ForegroundColor Yellow
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Python not found. Please install Python." -ForegroundColor Red
        exit 1
    }
    
    # Check Kaggle
    try {
        $kaggleVersion = kaggle --version 2>&1
        Write-Host "✓ Kaggle CLI installed" -ForegroundColor Green
    }
    catch {
        Write-Host "Installing Kaggle CLI..." -ForegroundColor Yellow
        pip install kaggle
    }
    
    # Check Kaggle credentials
    $kaggleJson = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
    if (Test-Path $kaggleJson) {
        Write-Host "✓ Kaggle credentials found" -ForegroundColor Green
    }
    else {
        Write-Host "⚠ Kaggle credentials not found at $kaggleJson" -ForegroundColor Yellow
        Write-Host "  Create API token at: https://www.kaggle.com/settings/account"
    }
    
    Write-Host ""
}

function Get-PlantVillage {
    Write-Host "--------------------------------------------" -ForegroundColor Blue
    Write-Host "📥 Downloading PlantVillage Dataset" -ForegroundColor Blue
    Write-Host "--------------------------------------------" -ForegroundColor Blue
    Write-Host "Source: https://www.kaggle.com/datasets/emmarex/plantdisease"
    Write-Host "Size: ~3.5 GB (87,000+ images)"
    Write-Host ""
    
    Set-Location $DiseasesDir
    
    if (Test-Path "PlantVillage" -or Test-Path "plantdisease") {
        Write-Host "PlantVillage already exists. Skipping." -ForegroundColor Yellow
        return
    }
    
    try {
        kaggle datasets download -d emmarex/plantdisease --unzip
        Write-Host "✓ PlantVillage downloaded successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to download. Manual download:" -ForegroundColor Red
        Write-Host "  https://www.kaggle.com/datasets/emmarex/plantdisease"
    }
    
    Set-Location $ScriptDir
}

function Get-DeepWeeds {
    Write-Host "--------------------------------------------" -ForegroundColor Blue
    Write-Host "📥 Downloading DeepWeeds Dataset" -ForegroundColor Blue
    Write-Host "--------------------------------------------" -ForegroundColor Blue
    Write-Host "Source: https://github.com/AlexOlsen/DeepWeeds"
    Write-Host ""
    
    Set-Location $WeedsDir
    
    if (Test-Path "DeepWeeds" -or Test-Path "deepweeds") {
        Write-Host "DeepWeeds already exists. Skipping." -ForegroundColor Yellow
        return
    }
    
    try {
        git clone --depth 1 https://github.com/AlexOlsen/DeepWeeds.git
        Write-Host "✓ DeepWeeds repository cloned" -ForegroundColor Green
        
        # Try Kaggle download for images
        kaggle datasets download -d imsparsh/deepweeds --unzip -p DeepWeeds/
    }
    catch {
        Write-Host "Clone failed. Try Kaggle:" -ForegroundColor Yellow
        try {
            kaggle datasets download -d imsparsh/deepweeds --unzip
            Write-Host "✓ DeepWeeds downloaded from Kaggle" -ForegroundColor Green
        }
        catch {
            Write-Host "Failed to download DeepWeeds" -ForegroundColor Red
        }
    }
    
    Set-Location $ScriptDir
}

function Get-Backgrounds {
    Write-Host "--------------------------------------------" -ForegroundColor Blue
    Write-Host "📥 Generating Background Images" -ForegroundColor Blue
    Write-Host "--------------------------------------------" -ForegroundColor Blue
    
    Set-Location $BackgroundsDir
    
    $existingCount = (Get-ChildItem -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
    if ($existingCount -gt 10) {
        Write-Host "Background images already exist ($existingCount images). Skipping." -ForegroundColor Yellow
        Set-Location $ScriptDir
        return
    }
    
    Write-Host "Generating synthetic field backgrounds..."
    
    $pythonScript = @"
import numpy as np
import cv2
import random

for i in range(20):
    base_green = random.randint(30, 80)
    base_brown = random.randint(40, 100)
    
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    if random.random() > 0.4:
        base_color = [base_green, base_green + 40, base_green - 10]
    else:
        base_color = [base_brown - 20, base_brown, base_brown + 20]
    
    for y in range(640):
        for x in range(640):
            noise = [random.randint(-20, 20) for _ in range(3)]
            img[y, x] = [max(0, min(255, base_color[c] + noise[c])) for c in range(3)]
    
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(f'synthetic_field_{i:02d}.jpg', img)

print('Generated 20 synthetic backgrounds')
"@
    
    $pythonScript | python
    
    Write-Host "✓ Background images ready" -ForegroundColor Green
    Set-Location $ScriptDir
}

function Get-IndianCrops {
    Write-Host "--------------------------------------------" -ForegroundColor Blue
    Write-Host "📥 Downloading Indian Crop Datasets" -ForegroundColor Blue
    Write-Host "--------------------------------------------" -ForegroundColor Blue
    
    Set-Location $TabularDir
    
    if (Test-Path "Crop_recommendation.csv") {
        Write-Host "Indian crop data exists. Skipping." -ForegroundColor Yellow
        Set-Location $ScriptDir
        return
    }
    
    try {
        kaggle datasets download -d atharvaingle/crop-recommendation-dataset --unzip
        Write-Host "✓ Indian crop datasets downloaded" -ForegroundColor Green
    }
    catch {
        Write-Host "Kaggle download failed. Generate synthetic data:" -ForegroundColor Yellow
        Write-Host "  python generate_agri_data.py"
    }
    
    Set-Location $ScriptDir
}

function Show-Stats {
    Write-Host "============================================" -ForegroundColor Blue
    Write-Host "📊 Dataset Statistics" -ForegroundColor Blue
    Write-Host "============================================" -ForegroundColor Blue
    
    Write-Host "`nDisease Images:" -ForegroundColor Green
    if (Test-Path $DiseasesDir) {
        $diseaseCount = (Get-ChildItem -Path $DiseasesDir -Recurse -Include "*.jpg","*.png" -ErrorAction SilentlyContinue).Count
        Write-Host "  Total images: $diseaseCount"
    }
    
    Write-Host "`nWeed Images:" -ForegroundColor Green
    if (Test-Path $WeedsDir) {
        $weedCount = (Get-ChildItem -Path $WeedsDir -Recurse -Include "*.jpg","*.png" -ErrorAction SilentlyContinue).Count
        Write-Host "  Total images: $weedCount"
    }
    
    Write-Host "`nBackground Images:" -ForegroundColor Green
    if (Test-Path $BackgroundsDir) {
        $bgCount = (Get-ChildItem -Path $BackgroundsDir -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
        Write-Host "  Total images: $bgCount"
    }
    
    Write-Host "`nTabular Data:" -ForegroundColor Green
    if (Test-Path $TabularDir) {
        Get-ChildItem -Path $TabularDir -Filter "*.csv" | ForEach-Object {
            $size = [math]::Round($_.Length / 1KB, 2)
            Write-Host "  $($_.Name): $size KB"
        }
    }
}

function Show-Help {
    Write-Host @"
AgriSense Dataset Downloader (Windows)

Usage: .\download_datasets.ps1 [OPTIONS]

Options:
  -All           Download all datasets
  -PlantVillage  Download PlantVillage (disease) dataset
  -DeepWeeds     Download DeepWeeds dataset
  -Backgrounds   Generate background images
  -IndianCrops   Download Indian crop tabular data
  -Stats         Show dataset statistics
  -Help          Show this help message

Examples:
  .\download_datasets.ps1 -All
  .\download_datasets.ps1 -PlantVillage -Backgrounds
"@
}

# Main
if ($Help) {
    Show-Help
    exit 0
}

Show-Header
Test-Dependencies

if ($All) {
    Get-PlantVillage
    Get-DeepWeeds
    Get-Backgrounds
    Get-IndianCrops
    Show-Stats
}
else {
    if ($PlantVillage) { Get-PlantVillage }
    if ($DeepWeeds) { Get-DeepWeeds }
    if ($Backgrounds) { Get-Backgrounds }
    if ($IndianCrops) { Get-IndianCrops }
    if ($Stats) { Show-Stats }
}

Write-Host "============================================" -ForegroundColor Green
Write-Host "✅ Dataset operations complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Generate tabular data: python generate_agri_data.py"
Write-Host "  2. Augment images: python augment_vision_data.py --demo"
Write-Host "  3. Train models: python train_all.py"
