@echo off
REM AgriSense Setup Script for Windows

echo 🌾 AgriSense Setup Starting...
echo ================================

REM Check Python
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python 3.12+ required
    exit /b 1
)

REM Check Node.js
echo Checking Node.js version...
node --version
if %errorlevel% neq 0 (
    echo ❌ Node.js 20+ required
    exit /b 1
)

REM Backend Setup
echo.
echo 📦 Setting up Backend...
cd backend

REM Create virtual environment
python -m venv .venv
call .venv\Scripts\activate.bat

REM Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Initialize database
python -c "from core.data_store import init_sensor_db; init_sensor_db()"

echo ✅ Backend setup complete!

REM Frontend Setup
echo.
echo 🎨 Setting up Frontend...
cd ..\frontend

REM Install dependencies
call npm install

echo ✅ Frontend setup complete!

echo.
echo ================================
echo ✅ Setup Complete!
echo.
echo To start the application:
echo 1. Backend:  cd backend && .venv\Scripts\activate && uvicorn main:app --reload
echo 2. Frontend: cd frontend && npm run dev
echo.
echo Access the app at: http://localhost:5173
echo API documentation: http://localhost:8000/docs

pause
