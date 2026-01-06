#!/usr/bin/env python3
"""
AgriSense Data Directory Setup Script
=====================================
Creates the standardized folder structure for ML datasets.

Directory Structure:
    src/backend/ml/data/
    ├── tabular/           (generated CSVs)
    ├── images/
    │   ├── diseases/      (raw disease images)
    │   ├── weeds/         (raw weed images)
    │   ├── backgrounds/   (field backgrounds for augmentation)
    │   └── augmented/     (processed output)
    └── intent_corpus/     (NLP training data)

Usage:
    python setup_data_dirs.py
    
Author: AgriSense ML Team
"""

import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_data_directories(base_path: str = None) -> dict:
    """
    Create the complete data directory structure for AgriSense ML pipeline.
    
    Args:
        base_path: Base path for data directory. If None, uses current script location.
        
    Returns:
        dict: Mapping of directory names to their absolute paths
    """
    if base_path is None:
        # Get the directory where this script is located
        base_path = Path(__file__).parent
    else:
        base_path = Path(base_path)
    
    # Define directory structure
    directories = {
        'tabular': base_path / 'tabular',
        'images': base_path / 'images',
        'diseases': base_path / 'images' / 'diseases',
        'weeds': base_path / 'images' / 'weeds',
        'backgrounds': base_path / 'images' / 'backgrounds',
        'augmented': base_path / 'images' / 'augmented',
        'augmented_diseases': base_path / 'images' / 'augmented' / 'diseases',
        'augmented_weeds': base_path / 'images' / 'augmented' / 'weeds',
        'intent_corpus': base_path / 'intent_corpus',
        'models': base_path / 'models',
        'models_tabular': base_path / 'models' / 'tabular',
        'models_vision': base_path / 'models' / 'vision',
        'models_nlp': base_path / 'models' / 'nlp',
        'models_edge': base_path / 'models' / 'edge',
    }
    
    # Create directories
    created_dirs = {}
    for name, path in directories.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            created_dirs[name] = str(path.absolute())
            logger.info(f"✓ Created: {path}")
        except Exception as e:
            logger.error(f"✗ Failed to create {path}: {e}")
            raise
    
    # Create .gitkeep files to preserve empty directories in git
    for name, path in directories.items():
        gitkeep = Path(path) / '.gitkeep'
        if not any(Path(path).iterdir()):
            gitkeep.touch()
            logger.debug(f"  Added .gitkeep to {name}")
    
    # Create README files for documentation
    readme_contents = {
        'tabular': """# Tabular Data Directory

Contains generated CSV datasets for crop recommendation and yield prediction models.

## Files:
- `india_crops_complete.csv` - Main crop recommendation dataset (22 crops × 2000 samples)
- `historical_yields.csv` - Yield prediction dataset with temporal features

## Schema:
### india_crops_complete.csv
| Column | Type | Description |
|--------|------|-------------|
| N | float | Nitrogen content (kg/ha) |
| P | float | Phosphorus content (kg/ha) |
| K | float | Potassium content (kg/ha) |
| temperature | float | Temperature (°C) |
| humidity | float | Relative humidity (%) |
| ph | float | Soil pH |
| rainfall | float | Annual rainfall (mm) |
| soil_type | str | Soil classification |
| label | str | Crop name |

### historical_yields.csv
Additional columns: year, pest_incidence, fertilizer_usage_kg, yield_t_ha
""",
        'images': """# Images Directory

Contains raw and augmented image datasets for vision models.

## Subdirectories:
- `diseases/` - Raw disease images (PlantVillage format)
- `weeds/` - Raw weed images (DeepWeeds format)
- `backgrounds/` - Field background images for augmentation
- `augmented/` - Processed images with Copy-Paste augmentation
""",
        'intent_corpus': """# Intent Corpus Directory

Contains training data for NLP models (chatbot intent classification).

## Files:
- `intents.json` - Intent classification training data
- `agricultural_qa.json` - Question-answer pairs for RAG
- `multilingual_queries.json` - Hindi/Tamil/English query examples
"""
    }
    
    for dir_name, content in readme_contents.items():
        readme_path = directories[dir_name] / 'README.md'
        if not readme_path.exists():
            readme_path.write_text(content)
            logger.info(f"  Added README.md to {dir_name}")
    
    return created_dirs


def print_tree(base_path: Path, prefix: str = ""):
    """Print directory tree structure."""
    contents = sorted(base_path.iterdir())
    pointers = ['├── '] * (len(contents) - 1) + ['└── ']
    
    for pointer, path in zip(pointers, contents):
        print(f"{prefix}{pointer}{path.name}{'/' if path.is_dir() else ''}")
        if path.is_dir():
            extension = '│   ' if pointer == '├── ' else '    '
            print_tree(path, prefix + extension)


def main():
    """Main entry point."""
    print("=" * 60)
    print("🌱 AgriSense Data Directory Setup")
    print("=" * 60)
    
    # Setup directories
    dirs = setup_data_directories()
    
    print("\n" + "=" * 60)
    print("📂 Created Directory Structure:")
    print("=" * 60)
    
    base = Path(__file__).parent
    print_tree(base)
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python generate_agri_data.py")
    print("  2. Run: bash download_datasets.sh")
    print("  3. Run: python augment_vision_data.py")
    
    return dirs


if __name__ == "__main__":
    main()
