@echo off
REM AgriSense Local-First Backend Quick Start (Windows)
REM This script sets up everything locally and prepares for Firebase deployment

setlocal enabledelayedexpansion

echo.
echo ========================================
echo AgriSense Local-First Backend Setup
echo ========================================
echo.

REM Check Node.js
echo [1/5] Checking Node.js...
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js not found. Please install Node.js 16+
    exit /b 1
)
for /f "tokens=*" %%i in ('node -v') do set NODE_VERSION=%%i
echo [OK] Node.js %NODE_VERSION%

REM Check npm
echo [2/5] Checking npm...
where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: npm not found
    exit /b 1
)
for /f "tokens=*" %%i in ('npm -v') do set NPM_VERSION=%%i
echo [OK] npm %NPM_VERSION%

REM Install PouchDB server dependencies
echo.
echo [3/5] Installing PouchDB Server dependencies...
call npm install express cors pouchdb body-parser
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install PouchDB dependencies
    exit /b 1
)
echo [OK] PouchDB dependencies installed

REM Install Frontend dependencies
echo.
echo [4/5] Installing Frontend dependencies...
cd src\frontend
call npm install pouchdb pouchdb-adapter-idb firebase
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install frontend dependencies
    cd ..\..
    exit /b 1
)

REM Create .env.local
echo.
echo [Creating frontend environment file...]
(
    echo VITE_POUCHDB_SERVER_URL=http://localhost:5984
    echo VITE_BACKEND_API_URL=http://localhost:8004/api/v1
    echo VITE_BACKEND_WS_URL=ws://localhost:8004
    echo VITE_ENABLE_OFFLINE_MODE=true
    echo VITE_ENABLE_POUCHDB_SYNC=true
    echo VITE_LOG_LEVEL=info
) > .env.local
echo [OK] Created .env.local

cd ..\..

REM Check Firebase CLI
echo.
echo [5/5] Checking Firebase CLI...
where firebase >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Firebase CLI not found
    echo Run this to install Firebase CLI:
    echo   npm install -g firebase-tools
    echo.
) else (
    echo [OK] Firebase CLI installed
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Start PouchDB Backend (Command Prompt 1):
echo    node pouchdb-server.js
echo.
echo 2. Start FastAPI Backend (Command Prompt 2):
echo    cd src\backend
echo    python -m uvicorn main:app --reload --port 8004
echo.
echo 3. Start Frontend Development Server (Command Prompt 3):
echo    cd src\frontend
echo    npm run dev
echo.
echo 4. Access the application:
echo    http://localhost:5173
echo.
echo 5. For Firebase deployment:
echo    firebase login
echo    cd src\frontend
echo    npm run build
echo    firebase deploy --only hosting
echo.
echo See FIREBASE_POUCHDB_DEPLOYMENT.md for detailed instructions
echo.
pause
