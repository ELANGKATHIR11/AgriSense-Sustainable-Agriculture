#!/bin/bash
# Setup script for Claude Crop Recommender System
# For Linux/Mac users

set -e  # Exit on any error

echo "========================================"
echo "Claude Crop Recommender - Setup Script"
echo "========================================"
echo ""

# Check Python version
echo "Step 1: Checking Python version..."
python_version=$(python --version 2>&1 | grep -oP 'Python \d+\.\d+')
echo "✓ Found: $python_version"
echo ""

# Create virtual environment (optional but recommended)
echo "Step 2: Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  • Virtual environment already exists"
else
    python -m venv venv
    echo "  ✓ Created virtual environment"
fi

# Activate virtual environment
echo "  • Activating virtual environment..."
source venv/bin/activate
echo ""

# Install dependencies
echo "Step 3: Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Run validation
echo "Step 4: Validating installation..."
python validate_setup.py
echo ""

# Summary
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Start service: uvicorn routes:router --port 8000"
echo "  3. Test it: curl http://localhost:8000/crop-recommendation/health"
echo ""
