"""
Automated Dataset Downloader for AgriSense ML Enhancement
Downloads compressed datasets for all ML models (8GB optimized version)
"""

import json
import sys
from pathlib import Path

# Base directory for ML
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets_enhanced"
DATASETS_DIR.mkdir(exist_ok=True)

print("🌾 AgriSense Dataset Downloader (8GB Optimized)")
print("=" * 60)

# Dataset URLs (Note: Some require manual Kaggle login)
DATASETS = {
    "crop_recommendation": {
        "name": "Smart Farming 2024 (Sample)",
        "size": "20 MB",
        "manual": True,
        "instructions": """
        1. Go to: https://www.kaggle.com/datasets/
        2. Search for "Smart Farming 2024" or "Crop Recommendation"
        3. Download and extract to: datasets_enhanced/crop_recommendation/
        """,
        "required_files": ["crop_data.csv"],
    },
    "yield_prediction": {
        "name": "Agriculture Crop Yield (Sample)",
        "size": "50 MB",
        "manual": True,
        "instructions": """
        1. Go to: https://www.kaggle.com/datasets/
        2. Search for "Agriculture Crop Yield"
        3. Download sample (100K records) to: datasets_enhanced/yield_prediction/
        """,
        "required_files": ["yield_data.csv"],
    },
    "water_requirement": {
        "name": "Water Requirement Dataset",
        "size": "30 MB",
        "manual": True,
        "instructions": """
        1. Can reuse crop recommendation data with water columns
        2. Or search Kaggle for "Irrigation" or "Water Requirement"
        3. Save to: datasets_enhanced/water_requirement/
        """,
        "required_files": ["water_data.csv"],
    },
    "season_classification": {
        "name": "Season Classification Dataset",
        "size": "20 MB",
        "manual": True,
        "instructions": """
        1. Can derive from crop recommendation (season column)
        2. Or search for "Agricultural Season" datasets
        3. Save to: datasets_enhanced/season_classification/
        """,
        "required_files": ["season_data.csv"],
    },
    "crop_type": {
        "name": "Crop Type Classification",
        "size": "30 MB",
        "manual": True,
        "instructions": """
        1. Use PlantVillage metadata or crop categorization data
        2. Save to: datasets_enhanced/crop_type/
        """,
        "required_files": ["crop_type_data.csv"],
    },
    "plant_disease": {
        "name": "PlantVillage Disease Detection (Grayscale)",
        "size": "500 MB",
        "url": "https://github.com/spMohanty/PlantVillage-Dataset",
        "manual": True,
        "instructions": """
        1. Clone: git clone https://github.com/spMohanty/PlantVillage-Dataset
        2. Use grayscale version (PlantVillage-Dataset/raw/grayscale/)
        3. Extract to: datasets_enhanced/plant_disease/
        4. Limit to ~20,000 core images to save space
        """,
        "required_files": ["images/"],
    },
    "agricultural_qa": {
        "name": "Agricultural Q&A for Chatbot",
        "size": "100 MB",
        "manual": True,
        "instructions": """
        1. Download from: https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only
        2. Or search Kaggle for "Agricultural Q&A" or "Farming Knowledge Base"
        3. Save to: datasets_enhanced/agricultural_qa/
        4. Need ~5,000 high-quality Q&A pairs
        """,
        "required_files": ["qa_data.json", "qa_data.csv"],
    },
}


def check_manual_download_status():
    """Check which datasets still need manual download"""
    print("\n📦 Checking Dataset Status...")
    print("-" * 60)

    needs_download = []
    completed = []

    for key, info in DATASETS.items():
        dataset_path = DATASETS_DIR / key
        if dataset_path.exists() and any(dataset_path.glob("*")):
            completed.append(key)
            print(f"✅ {info['name']}: Ready ({info['size']})")
        else:
            needs_download.append(key)
            print(f"❌ {info['name']}: Needs download ({info['size']})")

    return needs_download, completed


def print_download_instructions(dataset_keys):
    """Print manual download instructions"""
    if not dataset_keys:
        print("\n🎉 All datasets are ready!")
        return

    print("\n" + "=" * 60)
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("\n⚠️  Most datasets require Kaggle account (free)")
    print("📝 Please download the following datasets:\n")

    for key in dataset_keys:
        info = DATASETS[key]
        print(f"\n{'='*60}")
        print(f"Dataset: {info['name']} ({info['size']})")
        print(f"{'='*60}")
        print(info["instructions"])
        print(f"\nSave to: {DATASETS_DIR / key}/")
        print(
            f"Required files: {', '.join(info.get('required_files', ['any data files']))}"
        )

    print("\n" + "=" * 60)
    print("After downloading, run this script again to verify!")
    print("=" * 60)


def create_sample_data_if_needed():
    """Create minimal sample data for testing if datasets not downloaded"""
    print("\n🔧 Creating sample data for testing...")

    # Create crop recommendation sample
    crop_rec_dir = DATASETS_DIR / "crop_recommendation"
    crop_rec_dir.mkdir(exist_ok=True)
    if not (crop_rec_dir / "crop_data.csv").exists():
        sample_csv = crop_rec_dir / "crop_data.csv"
        with open(sample_csv, "w") as f:
            f.write("N,P,K,temperature,humidity,ph,rainfall,label\n")
            f.write("90,42,43,20.87,82.00,6.50,202.93,rice\n")
            f.write("85,58,41,21.77,80.31,7.03,226.65,rice\n")
            f.write("60,55,44,23.00,82.31,7.84,263.96,rice\n")
        print(f"✅ Created sample: {sample_csv}")

    print(
        "ℹ️  Note: Sample data is for testing only. Download full datasets for production!"
    )


def main():
    print("\n🚀 Starting Dataset Download Check...\n")

    # Check status
    needs_download, completed = check_manual_download_status()

    # Print summary
    print("\n📊 Summary:")
    print(f"   ✅ Ready: {len(completed)}/7 datasets")
    print(f"   ❌ Needed: {len(needs_download)}/7 datasets")

    if needs_download:
        print_download_instructions(needs_download)

        # Ask if user wants to create sample data
        print("\n" + "=" * 60)
        print("Would you like to create sample data for testing? (y/n)")
        print("(You can still download full datasets later)")
        print("=" * 60)

        create_sample_data_if_needed()
    else:
        print("\n✅ All datasets ready! You can now run model retraining.")
        print(f"\n📁 Datasets location: {DATASETS_DIR.absolute()}")
        print("\nNext steps:")
        print("1. Run: python retrain_all_models.py")
        print("2. Run: python train_disease_detection.py")
        print("3. Run: python download_phi2.py")
        print("4. Run: python finetune_phi2_agriculture.py")

    # Save status
    status_file = BASE_DIR / "dataset_status.json"
    with open(status_file, "w") as f:
        json.dump(
            {
                "completed": completed,
                "needs_download": needs_download,
                "total": len(DATASETS),
            },
            f,
            indent=2,
        )

    print(f"\n💾 Status saved to: {status_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download check cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
