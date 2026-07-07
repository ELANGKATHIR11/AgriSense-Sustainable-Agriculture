# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

$env:CI = "true"
$env:PATH = "F:\FULL-STACK;" + $env:PATH

$root = $PSScriptRoot
if (-not $root) { $root = Get-Location }

$pythonPip = "C:\Users\elang\Miniconda3\envs\dgpu-core\Scripts\pip.exe"
$pyinstallerExe = "C:\Users\elang\Miniconda3\envs\dgpu-core\Scripts\pyinstaller.exe"
$nodeExe = "F:\FULL-STACK\node.exe"

Write-Host "📦 1. Installing PyInstaller inside Python Environment..." -ForegroundColor Green
& $pythonPip install pyinstaller

Write-Host "📦 2. Installing Electron & Electron-Builder packages (ignoring scripts)..." -ForegroundColor Green
npm install --save-dev electron electron-builder --ignore-scripts

Write-Host "📥 3. Downloading Electron Binaries manually..." -ForegroundColor Green
if (Test-Path "node_modules/electron/install.js") {
    & $nodeExe node_modules/electron/install.js
}

Write-Host "🔨 4. Compiling React frontend..." -ForegroundColor Green
& $nodeExe node_modules/vite/bin/vite.js build
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Vite build failed."
    exit 1
}

& $nodeExe node_modules/esbuild/bin/esbuild server.ts --bundle --platform=node --format=cjs --packages=external --sourcemap --outfile=dist/server.cjs
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Esbuild server compile failed."
    exit 1
}

Write-Host "🐍 5. Bundling Python FastAPI + ML backend using PyInstaller..." -ForegroundColor Green
if (-not (Test-Path $pyinstallerExe)) {
    Write-Warning "PyInstaller not found at $pyinstallerExe. Trying system fallback..."
    $pyinstallerExe = "pyinstaller"
}

& $pyinstallerExe --noconfirm --onedir --console --name "agrisense-backend" --add-data "backend;backend" --add-data "ml;ml" --distpath "backend-dist" backend/main.py

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ PyInstaller backend compilation failed. Aborting."
    exit 1
}

Write-Host "⚙️ 6. Packaging Electron Desktop Application..." -ForegroundColor Green
& $nodeExe node_modules/electron-builder/out/cli/cli.js

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Electron packaging failed."
    exit 1
}

Write-Host "✨ 7. Complete! Your desktop app installer is saved in:" -ForegroundColor Cyan
Write-Host "   $root\dist-desktop\AgriSense Setup 0.0.0.exe" -ForegroundColor Green
