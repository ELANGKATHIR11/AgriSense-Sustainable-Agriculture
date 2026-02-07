#!/usr/bin/env python
"""
Validation script for Claude Crop Recommender System
Checks that all dependencies are installed and code loads correctly
"""

import sys
import importlib


def check_python_version():
    """Verify Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Verify all required packages are installed"""
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib',
        'fastapi': 'fastapi',
        'pydantic': 'pydantic',
    }

    missing = []
    for import_name, package_name in required.items():
        try:
            importlib.import_module(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - NOT INSTALLED")
            missing.append(package_name)

    return len(missing) == 0, missing


def check_module_imports():
    """Verify core modules can be imported"""
    print("\nChecking module imports...")
    modules = [
        'crop_requirements_dataset',
        'crop_recommendation_ml_model',
        'crop_recommender_api',
        'routes',
    ]

    all_ok = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"❌ {module} - {str(e)}")
            all_ok = False

    return all_ok


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("Claude Crop Recommender - Validation Check")
    print("=" * 60)

    print("\nStep 1: Checking Python version...")
    if not check_python_version():
        sys.exit(1)

    print("\nStep 2: Checking dependencies...")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Fix by running: pip install -r requirements.txt")
        sys.exit(1)

    print("\nStep 3: Checking module imports...")
    if not check_module_imports():
        print("\n❌ Some modules failed to import")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ All checks passed!")
    print("=" * 60)
    print("\nThe system is ready to use:")
    print(
        "  • Use as FastAPI service: "
        "uvicorn routes:router --port 8000"
    )
    print("  • Use as Python module:")
    print("    from crop_recommender_api import recommend_crops")
    print("  • Run test: python crop_recommender_api.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
