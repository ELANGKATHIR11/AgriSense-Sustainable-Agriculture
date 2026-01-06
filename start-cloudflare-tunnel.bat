@echo off
REM Start Cloudflare Tunnel for AgriSense Backend

setlocal enabledelayedexpansion

set CLOUDFLARED=%TEMP%\cloudflared.exe
set CONFIG_DIR=%USERPROFILE%\.cloudflared
set CONFIG=%CONFIG_DIR%\config.yml

echo.
echo ============================================
echo  Starting AgriSense Backend Tunnel
echo ============================================
echo.

REM Check if certificate exists
if not exist "%CONFIG_DIR%\cert.pem" (
    echo ERROR: Not authenticated with Cloudflare
    echo Please run setup-cloudflare-tunnel.bat first
    pause
    exit /b 1
)

REM Start the tunnel
echo Starting Cloudflare Tunnel...
echo Tunnel: agrisense-api
echo Backend: http://localhost:8004
echo.
echo Press CTRL+C to stop the tunnel
echo.

cd /d "%USERPROFILE%\.cloudflared"
"%CLOUDFLARED%" tunnel run agrisense-api

pause
