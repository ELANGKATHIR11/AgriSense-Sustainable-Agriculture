@echo off
echo Starting AgriSense Backend with fixed imports...
cd /d "%~dp0\src\backend"
python start_fixed.py
if errorlevel 1 (
    echo Server crashed or exited with error.
)
pause
