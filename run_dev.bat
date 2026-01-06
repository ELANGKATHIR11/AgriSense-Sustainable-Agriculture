@echo off
echo Starting AgriSense in DEV mode...

echo Starting Backend...
start "AgriSense Backend" cmd /k "cd src\backend && python start_fixed.py"

echo Starting Frontend...
cd src\frontend
echo Installing dependencies if needed...
if not exist node_modules call npm install
echo Starting Vite...
start "AgriSense Frontend" cmd /k "npm run dev"

echo.
echo AgriSense is starting!
echo Backend: http://localhost:8004
echo Frontend: http://localhost:5173 (usually)
echo.
pause
