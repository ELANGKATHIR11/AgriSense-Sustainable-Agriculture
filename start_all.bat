@echo off
setlocal
set "ROOT=%~dp0"
set "BACKEND_PY=C:\Users\elang\Miniconda3\envs\dgpu-core\python.exe"

if not exist "%BACKEND_PY%" (
  echo Backend Python not found: "%BACKEND_PY%"
  exit /b 1
)

if not exist "%ROOT%node_modules" (
  echo node_modules is missing. Run npm install first.
  exit /b 1
)

cd /d "%ROOT%"

echo Starting backend on http://localhost:8000 ...
start "AgriSense Backend" cmd /k ""%BACKEND_PY%" -u -c "import backend.main as appmod; import uvicorn; uvicorn.run(appmod.app, host='0.0.0.0', port=8000, log_level='info')""

echo Starting frontend on http://localhost:3000 ...
start "AgriSense Frontend" cmd /k "cd /d ""%ROOT%"" && npm run dev -- --host 0.0.0.0 --port 3000"

echo Both servers are launching in separate windows.
