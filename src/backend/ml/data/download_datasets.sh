#!/bin/bash
#
# AgriSense Dataset Download Helper
# ==================================
# Downloads external datasets required for training vision models.
#
# Prerequisites:
#   - Kaggle CLI: pip install kaggle
#   - Kaggle API token: ~/.kaggle/kaggle.json
#   - wget, unzip
#
# Usage:
#   chmod +x download_datasets.sh
#   ./download_datasets.sh [--all|--plantvillage|--deepweeds|--backgrounds]
#
# Author: AgriSense ML Team

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory (script location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${SCRIPT_DIR}"
IMAGES_DIR="${DATA_DIR}/images"
DISEASES_DIR="${IMAGES_DIR}/diseases"
WEEDS_DIR="${IMAGES_DIR}/weeds"
BACKGROUNDS_DIR="${IMAGES_DIR}/backgrounds"

# Create directories
mkdir -p "${DISEASES_DIR}" "${WEEDS_DIR}" "${BACKGROUNDS_DIR}"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}🌾 AgriSense Dataset Downloader${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Data directory: ${DATA_DIR}"
echo ""

# Check for required tools
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python is required but not installed.${NC}"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null; then
        echo -e "${YELLOW}Warning: pip not found. Some downloads may fail.${NC}"
    fi
    
    # Check wget
    if ! command -v wget &> /dev/null; then
        echo -e "${YELLOW}Warning: wget not found. Install with: apt install wget${NC}"
    fi
    
    # Check unzip
    if ! command -v unzip &> /dev/null; then
        echo -e "${YELLOW}Warning: unzip not found. Install with: apt install unzip${NC}"
    fi
    
    # Check kaggle
    if ! command -v kaggle &> /dev/null; then
        echo -e "${YELLOW}Kaggle CLI not found. Installing...${NC}"
        pip install kaggle
    fi
    
    # Check kaggle credentials
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo -e "${RED}Warning: Kaggle credentials not found at ~/.kaggle/kaggle.json${NC}"
        echo -e "Please create a Kaggle API token at: https://www.kaggle.com/settings/account"
        echo -e "Download kaggle.json and place it in ~/.kaggle/"
        echo ""
    fi
    
    echo -e "${GREEN}✓ Dependencies checked${NC}"
    echo ""
}

# Download PlantVillage dataset (Disease Detection)
download_plantvillage() {
    echo -e "${BLUE}--------------------------------------------${NC}"
    echo -e "${BLUE}📥 Downloading PlantVillage Dataset${NC}"
    echo -e "${BLUE}--------------------------------------------${NC}"
    echo "Source: https://www.kaggle.com/datasets/emmarex/plantdisease"
    echo "Size: ~3.5 GB (87,000+ images)"
    echo "Classes: 38 disease categories"
    echo ""
    
    cd "${DISEASES_DIR}"
    
    if [ -d "PlantVillage" ] || [ -d "plantvillage" ]; then
        echo -e "${YELLOW}PlantVillage already exists. Skipping download.${NC}"
        return 0
    fi
    
    # Method 1: Kaggle CLI
    if command -v kaggle &> /dev/null && [ -f ~/.kaggle/kaggle.json ]; then
        echo "Using Kaggle CLI..."
        kaggle datasets download -d emmarex/plantdisease --unzip
        echo -e "${GREEN}✓ PlantVillage downloaded successfully${NC}"
    else
        # Method 2: Direct download (backup)
        echo -e "${YELLOW}Kaggle CLI not available. Trying alternative sources...${NC}"
        
        # Alternative: GitHub mirror (smaller subset)
        echo "Downloading PlantVillage subset from GitHub..."
        if command -v git &> /dev/null; then
            git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git PlantVillage_raw
            mv PlantVillage_raw/raw/color/* . 2>/dev/null || true
            rm -rf PlantVillage_raw
            echo -e "${GREEN}✓ PlantVillage (GitHub subset) downloaded${NC}"
        else
            echo -e "${RED}Cannot download PlantVillage. Please install Kaggle CLI or Git.${NC}"
            echo "Manual download: https://www.kaggle.com/datasets/emmarex/plantdisease"
            return 1
        fi
    fi
    
    echo ""
}

# Download DeepWeeds dataset (Weed Detection)
download_deepweeds() {
    echo -e "${BLUE}--------------------------------------------${NC}"
    echo -e "${BLUE}📥 Downloading DeepWeeds Dataset${NC}"
    echo -e "${BLUE}--------------------------------------------${NC}"
    echo "Source: https://github.com/AlexOlsen/DeepWeeds"
    echo "Paper: https://arxiv.org/abs/1811.04129"
    echo "Size: ~1.5 GB (17,509 images)"
    echo "Classes: 9 weed species"
    echo ""
    
    cd "${WEEDS_DIR}"
    
    if [ -d "DeepWeeds" ] || [ -d "deepweeds" ] || [ -d "images" ]; then
        echo -e "${YELLOW}DeepWeeds already exists. Skipping download.${NC}"
        return 0
    fi
    
    # Clone repository
    if command -v git &> /dev/null; then
        git clone --depth 1 https://github.com/AlexOlsen/DeepWeeds.git
        echo -e "${GREEN}✓ DeepWeeds repository cloned${NC}"
        
        # Download images from Kaggle (official source)
        echo "Downloading DeepWeeds images from Kaggle..."
        if command -v kaggle &> /dev/null && [ -f ~/.kaggle/kaggle.json ]; then
            kaggle datasets download -d imsparsh/deepweeds --unzip -p DeepWeeds/
        else
            # Alternative: Direct download from official source
            echo "Downloading from official source..."
            wget -c https://cloudstor.aarnet.edu.au/plus/s/uBWBJXsSRkXLSrQ/download -O deepweeds_images.zip
            unzip deepweeds_images.zip -d DeepWeeds/
            rm deepweeds_images.zip
        fi
        
        echo -e "${GREEN}✓ DeepWeeds downloaded successfully${NC}"
    else
        # Fallback: Kaggle only
        if command -v kaggle &> /dev/null && [ -f ~/.kaggle/kaggle.json ]; then
            kaggle datasets download -d imsparsh/deepweeds --unzip
            echo -e "${GREEN}✓ DeepWeeds downloaded from Kaggle${NC}"
        else
            echo -e "${RED}Cannot download DeepWeeds. Please install Git or Kaggle CLI.${NC}"
            return 1
        fi
    fi
    
    echo ""
}

# Download field background images for augmentation
download_backgrounds() {
    echo -e "${BLUE}--------------------------------------------${NC}"
    echo -e "${BLUE}📥 Downloading Field Background Images${NC}"
    echo -e "${BLUE}--------------------------------------------${NC}"
    echo "Source: Unsplash/Pexels (agriculture field images)"
    echo ""
    
    cd "${BACKGROUNDS_DIR}"
    
    existing_count=$(ls -1 *.jpg 2>/dev/null | wc -l)
    if [ "$existing_count" -gt "10" ]; then
        echo -e "${YELLOW}Background images already exist (${existing_count} images). Skipping.${NC}"
        return 0
    fi
    
    echo "Downloading sample field backgrounds..."
    
    # Sample URLs from free image sources (agriculture/field themed)
    # In production, replace with actual field images
    BACKGROUND_URLS=(
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=640"  # Green field
        "https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=640"  # Agriculture
        "https://images.unsplash.com/photo-1500595046743-cd271d694d30?w=640"  # Crops
        "https://images.unsplash.com/photo-1501004318641-b39e6451bec6?w=640"  # Plants
        "https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?w=640"  # Field
    )
    
    count=0
    for url in "${BACKGROUND_URLS[@]}"; do
        count=$((count + 1))
        filename="field_background_${count}.jpg"
        if [ ! -f "$filename" ]; then
            echo "  Downloading background ${count}..."
            wget -q -O "$filename" "$url" 2>/dev/null || curl -s -o "$filename" "$url" 2>/dev/null || true
        fi
    done
    
    # Generate synthetic backgrounds as fallback
    echo "Generating synthetic backgrounds..."
    python3 << 'EOF'
import numpy as np
import cv2
import os
import random

output_dir = "."
for i in range(15):
    # Create random field-like background
    base_green = random.randint(30, 80)
    base_brown = random.randint(40, 100)
    
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Choose between green field and brown soil
    if random.random() > 0.4:
        base_color = [base_green, base_green + 40, base_green - 10]  # Green
    else:
        base_color = [base_brown - 20, base_brown, base_brown + 20]  # Brown
    
    for y in range(640):
        for x in range(640):
            noise = [random.randint(-20, 20) for _ in range(3)]
            img[y, x] = [max(0, min(255, base_color[c] + noise[c])) for c in range(3)]
    
    # Add texture
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    cv2.imwrite(f"{output_dir}/synthetic_field_{i:02d}.jpg", img)

print(f"Generated 15 synthetic backgrounds")
EOF
    
    echo -e "${GREEN}✓ Background images ready${NC}"
    echo ""
}

# Download Indian Crop dataset from Kaggle
download_indian_crops() {
    echo -e "${BLUE}--------------------------------------------${NC}"
    echo -e "${BLUE}📥 Downloading Indian Crop Datasets${NC}"
    echo -e "${BLUE}--------------------------------------------${NC}"
    
    cd "${DATA_DIR}/tabular"
    
    if [ -f "Crop_recommendation.csv" ]; then
        echo -e "${YELLOW}Indian crop data already exists. Skipping.${NC}"
        return 0
    fi
    
    if command -v kaggle &> /dev/null && [ -f ~/.kaggle/kaggle.json ]; then
        # Popular crop recommendation dataset
        echo "Downloading crop recommendation dataset..."
        kaggle datasets download -d atharvaingle/crop-recommendation-dataset --unzip 2>/dev/null || true
        
        echo -e "${GREEN}✓ Indian crop datasets downloaded${NC}"
    else
        echo -e "${YELLOW}Kaggle not available. Generate synthetic data instead:${NC}"
        echo "  python generate_agri_data.py"
    fi
    
    echo ""
}

# Print dataset statistics
print_stats() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}📊 Dataset Statistics${NC}"
    echo -e "${BLUE}============================================${NC}"
    
    echo -e "\n${GREEN}Disease Images:${NC}"
    if [ -d "${DISEASES_DIR}" ]; then
        disease_count=$(find "${DISEASES_DIR}" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.JPG" \) 2>/dev/null | wc -l)
        echo "  Total images: ${disease_count}"
        echo "  Categories:"
        ls -d "${DISEASES_DIR}"/*/ 2>/dev/null | head -10 | while read dir; do
            count=$(find "$dir" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
            echo "    - $(basename "$dir"): ${count} images"
        done
    else
        echo "  Not downloaded"
    fi
    
    echo -e "\n${GREEN}Weed Images:${NC}"
    if [ -d "${WEEDS_DIR}" ]; then
        weed_count=$(find "${WEEDS_DIR}" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
        echo "  Total images: ${weed_count}"
    else
        echo "  Not downloaded"
    fi
    
    echo -e "\n${GREEN}Background Images:${NC}"
    if [ -d "${BACKGROUNDS_DIR}" ]; then
        bg_count=$(find "${BACKGROUNDS_DIR}" -type f -name "*.jpg" 2>/dev/null | wc -l)
        echo "  Total images: ${bg_count}"
    else
        echo "  Not downloaded"
    fi
    
    echo -e "\n${GREEN}Tabular Data:${NC}"
    if [ -d "${DATA_DIR}/tabular" ]; then
        ls -lh "${DATA_DIR}/tabular"/*.csv 2>/dev/null || echo "  No CSV files yet"
    fi
    
    echo ""
}

# Main menu
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all           Download all datasets"
    echo "  --plantvillage  Download PlantVillage (disease) dataset"
    echo "  --deepweeds     Download DeepWeeds dataset"
    echo "  --backgrounds   Download/generate background images"
    echo "  --indian-crops  Download Indian crop tabular data"
    echo "  --stats         Show dataset statistics"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all"
    echo "  $0 --plantvillage --backgrounds"
    echo ""
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

check_dependencies

for arg in "$@"; do
    case $arg in
        --all)
            download_plantvillage
            download_deepweeds
            download_backgrounds
            download_indian_crops
            print_stats
            ;;
        --plantvillage)
            download_plantvillage
            ;;
        --deepweeds)
            download_deepweeds
            ;;
        --backgrounds)
            download_backgrounds
            ;;
        --indian-crops)
            download_indian_crops
            ;;
        --stats)
            print_stats
            ;;
        --help)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✅ Dataset download complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Generate tabular data: python generate_agri_data.py"
echo "  2. Augment images: python augment_vision_data.py --demo"
echo "  3. Train models: python -m ml_v2.training.train_all"
echo ""
