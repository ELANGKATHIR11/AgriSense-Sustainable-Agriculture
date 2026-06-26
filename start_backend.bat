@echo off
setlocal
set "ROOT=%~dp0"
set "BACKEND_PY=C:\Users\elang\Miniconda3\envs\dgpu-core\python.exe"

if not exist "%BACKEND_PY%" (
  echo Backend Python not found: "%BACKEND_PY%"
  exit /b 1
)

cd /d "%ROOT%"

echo Starting FastAPI server on http://localhost:8000 ...
"%BACKEND_PY%" -u -c "import backend.main as appmod; import uvicorn; uvicorn.run(appmod.app, host='0.0.0.0', port=8000, log_level='info')"
