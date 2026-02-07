@echo off
REM Setup script for Claude Crop Recommender System
REM For Windows users

echo.
echo ========================================
echo Claude Crop Recommender - Setup Script
echo ========================================
echo.

REM Check Python version
echo Step 1: Checking Python version...
python --version
echo.

REM Create virtual environment
echo Step 2: Creating virtual environment...
if exist venv (
    echo   - Virtual environment already exists
) else (
    python -m venv venv
    echo   ✓ Created virtual environment
)
echo.

REM Activate virtual environment
echo Step 3: Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo Step 4: Installing Python dependencies...
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo ✓ Dependencies installed
echo.

REM Run validation
echo Step 5: Validating installation...
python validate_setup.py
echo.

REM Summary
echo ========================================
echo ✅ Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Activate environment: venv\Scripts\activate.bat
echo   2. Start service: uvicorn routes:router --port 8000
echo   3. Test it: curl http://localhost:8000/crop-recommendation/health
echo.
pause
