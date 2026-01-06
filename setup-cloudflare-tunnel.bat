@echo off
REM Cloudflare Tunnel Setup for AgriSense Backend

setlocal enabledelayedexpansion

set CLOUDFLARED=%TEMP%\cloudflared.exe
set CONFIG=%USERPROFILE%\.cloudflared\config.yml
set CERT_DIR=%USERPROFILE%\.cloudflared

REM Create certificate directory
if not exist "%CERT_DIR%" mkdir "%CERT_DIR%"

echo.
echo ============================================
echo  AgriSense Backend - Cloudflare Tunnel Setup
echo ============================================
echo.

REM Step 1: Check if authenticated
echo [1/3] Checking Cloudflare authentication...
if not exist "%CERT_DIR%\cert.pem" (
    echo Not authenticated. Opening browser for login...
    "%CLOUDFLARED%" login
    if !errorlevel! neq 0 (
        echo ERROR: Authentication failed
        echo Please visit: https://dash.cloudflare.com/
        echo And authorize cloudflared
        pause
        exit /b 1
    )
)
echo ✓ Authenticated with Cloudflare

REM Step 2: Create tunnel
echo.
echo [2/3] Creating tunnel...
"%CLOUDFLARED%" tunnel create agrisense-api 2>nul
if !errorlevel! neq 0 (
    echo Tunnel may already exist, continuing...
)
echo ✓ Tunnel created/verified

REM Step 3: Get tunnel ID and display public URL
echo.
echo [3/3] Getting tunnel information...
for /f %%A in ('"%CLOUDFLARED%" tunnel list ^| find "agrisense-api" ^| for /f "tokens=1" %%%%B in ('findstr "."') do @echo %%%%B') do set TUNNEL_ID=%%A

echo.
echo ============================================
echo ✅ Setup Complete!
echo ============================================
echo.
echo Your backend will be accessible at:
echo https://api-agrisense.local
echo.
echo To start the tunnel, run:
echo   "%CLOUDFLARED%" tunnel run agrisense-api
echo.
echo OR use the provided start script
echo.
pause
