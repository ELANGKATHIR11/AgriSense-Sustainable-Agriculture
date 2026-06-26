@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON_PIP=C:\Users\elang\Miniconda3\envs\dgpu-core\Scripts\pip.exe"
set "PYINSTALLER_EXE=C:\Users\elang\Miniconda3\envs\dgpu-core\Scripts\pyinstaller.exe"

cd /d "%ROOT%"

echo 📦 1. Installing PyInstaller inside Python Environment...
call "%PYTHON_PIP%" install pyinstaller

echo 📦 2. Installing Electron & Electron-Builder packages...
call npm install --save-dev electron electron-builder

echo 🔨 3. Compiling React frontend...
call npm run build

if %ERRORLEVEL% NEQ 0 (
  echo ❌ Frontend build failed. Aborting.
  exit /b 1
)

echo 🐍 4. Bundling Python FastAPI + ML backend using PyInstaller...
if not exist "%PYINSTALLER_EXE%" (
  echo ❌ PyInstaller not found at %PYINSTALLER_EXE%! Trying system fallback...
  set "PYINSTALLER_EXE=pyinstaller"
)

call "%PYINSTALLER_EXE%" --noconfirm --onedir --console --name "agrisense-backend" --add-data "backend;backend" --add-data "ml;ml" --distpath "backend-dist" backend/main.py

if %ERRORLEVEL% NEQ 0 (
  echo ❌ PyInstaller backend compilation failed. Aborting.
  exit /b 1
)

echo ⚙️ 5. Packaging Electron Desktop Application...
call npm run dist-pack

if %ERRORLEVEL% NEQ 0 (
  echo ❌ Electron packaging failed.
  exit /b 1
)

echo  6. Complete! Your downloadable desktop app installer is saved in:
echo      %ROOT%dist-desktop\AgriSense Setup 0.0.0.exe
pause
