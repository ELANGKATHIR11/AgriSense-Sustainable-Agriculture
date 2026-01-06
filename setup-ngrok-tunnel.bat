@echo off
REM Setup ngrok for backend tunneling
REM Run this after you have your ngrok authtoken

echo.
echo === AgriSense Backend ngrok Setup ===
echo.
echo Step 1: Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
echo.

set /p TOKEN="Paste your ngrok authtoken here: "

if "%TOKEN%"=="" (
    echo Error: Authtoken is required
    exit /b 1
)

echo.
echo Installing authtoken...
"%TEMP%\ngrok.exe" config add-authtoken %TOKEN%

echo.
echo Starting ngrok tunnel to localhost:8004...
echo The public URL will appear below - copy it to use with the frontend
echo.

"%TEMP%\ngrok.exe" http 8004
