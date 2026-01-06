# AgriSense Local-First Backend Setup (PowerShell)
# This script sets up everything for local-first development with Firebase

param(
    [switch]$SkipFirebase = $false
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AgriSense Local-First Backend Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Node.js
Write-Host "[1/5] Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node -v
    Write-Host "[OK] Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Node.js not found. Please install Node.js 16+" -ForegroundColor Red
    exit 1
}

# Check npm
Write-Host "[2/5] Checking npm..." -ForegroundColor Yellow
try {
    $npmVersion = npm -v
    Write-Host "[OK] npm $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] npm not found" -ForegroundColor Red
    exit 1
}

# Install PouchDB server dependencies
Write-Host ""
Write-Host "[3/5] Installing PouchDB Server dependencies..." -ForegroundColor Yellow
npm install express cors pouchdb body-parser
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install PouchDB dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] PouchDB dependencies installed" -ForegroundColor Green

# Install Frontend dependencies
Write-Host ""
Write-Host "[4/5] Installing Frontend dependencies..." -ForegroundColor Yellow
Push-Location src/frontend
npm install pouchdb pouchdb-adapter-idb firebase
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install frontend dependencies" -ForegroundColor Red
    Pop-Location
    exit 1
}

# Create .env.local
Write-Host ""
Write-Host "Creating frontend environment file..." -ForegroundColor Yellow
$envContent = @"
VITE_POUCHDB_SERVER_URL=http://localhost:5984
VITE_BACKEND_API_URL=http://localhost:8004/api/v1
VITE_BACKEND_WS_URL=ws://localhost:8004
VITE_ENABLE_OFFLINE_MODE=true
VITE_ENABLE_POUCHDB_SYNC=true
VITE_LOG_LEVEL=info
"@

$envContent | Out-File -FilePath ".env.local" -Encoding UTF8
Write-Host "[OK] Created .env.local" -ForegroundColor Green

Pop-Location

# Check Firebase CLI
Write-Host ""
Write-Host "[5/5] Checking Firebase CLI..." -ForegroundColor Yellow
try {
    $firebaseVersion = firebase --version
    Write-Host "[OK] Firebase CLI installed: $firebaseVersion" -ForegroundColor Green
} catch {
    if (-not $SkipFirebase) {
        Write-Host ""
        Write-Host "WARNING: Firebase CLI not found" -ForegroundColor Yellow
        Write-Host "Run this to install Firebase CLI:" -ForegroundColor Yellow
        Write-Host "  npm install -g firebase-tools" -ForegroundColor Cyan
        Write-Host ""
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Start PouchDB Backend (PowerShell 1):" -ForegroundColor Cyan
Write-Host "   node pouchdb-server.js" -ForegroundColor White
Write-Host ""
Write-Host "2. Start FastAPI Backend (PowerShell 2):" -ForegroundColor Cyan
Write-Host "   cd src\backend" -ForegroundColor White
Write-Host "   python -m uvicorn main:app --reload --port 8004" -ForegroundColor White
Write-Host ""
Write-Host "3. Start Frontend Development Server (PowerShell 3):" -ForegroundColor Cyan
Write-Host "   cd src\frontend" -ForegroundColor White
Write-Host "   npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "4. Access the application:" -ForegroundColor Cyan
Write-Host "   http://localhost:5173" -ForegroundColor White
Write-Host ""
Write-Host "5. For Firebase deployment:" -ForegroundColor Cyan
Write-Host "   firebase login" -ForegroundColor White
Write-Host "   cd src\frontend" -ForegroundColor White
Write-Host "   npm run build" -ForegroundColor White
Write-Host "   firebase deploy --only hosting" -ForegroundColor White
Write-Host ""
Write-Host "See FIREBASE_POUCHDB_DEPLOYMENT.md for detailed instructions" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"
