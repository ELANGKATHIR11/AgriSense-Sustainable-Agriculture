# AgriSense Full-Stack Startup with PocketDB
# This script starts PocketDB, Backend, and Frontend

param(
    [switch]$BackendOnly = $false,
    [switch]$FrontendOnly = $false,
    [switch]$NoPocketDB = $false,
    [string]$Backend = "pocketdb",
    [string]$Port = "8000"
)

$ErrorActionPreference = "Continue"

# Colors for output
function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "=" * 70 -ForegroundColor Blue
    Write-Host "  $Message" -ForegroundColor Green
    Write-Host "=" * 70 -ForegroundColor Blue
}

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "→ $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

# Main startup
Write-Header "AgriSense Full-Stack with PocketDB"

# Check virtual environment
Write-Section "Checking Virtual Environment"
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Error-Custom "Virtual environment not found at .venv"
    Write-Host "Please run: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}
Write-Success "Virtual environment found"

# Activate virtual environment
Write-Section "Activating Virtual Environment"
& ".venv\Scripts\Activate.ps1"
Write-Success "Virtual environment activated"

# Set environment variables
Write-Section "Setting Environment Variables"
$env:AGRISENSE_DB_BACKEND = $Backend
$env:POCKETDB_URL = "http://localhost:8090"
$env:POCKETDB_DATA_DIR = "./pb_data"
$env:VITE_API_URL = "http://localhost:8000"
$env:FASTAPI_ENV = "development"
$env:LOG_LEVEL = "INFO"

# Ensure pb_data directory exists
if (-not (Test-Path "pb_data")) {
    New-Item -ItemType Directory -Path "pb_data" -Force | Out-Null
    Write-Success "Created pb_data directory"
}

Write-Success "Environment variables set"

# Load .env.pocketdb if it exists
if (Test-Path ".env.pocketdb") {
    Write-Section "Loading Configuration from .env.pocketdb"
    Get-Content .env.pocketdb | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            $key = $matches[1]
            $value = $matches[2]
            if (-not (Test-Path "env:$key")) {
                Set-Item -Path "env:$key" -Value $value
            }
        }
    }
    Write-Success "Configuration loaded"
}

# Print startup info
Write-Header "AgriSense Startup Information"
Write-Host "Database Backend:   $Backend" -ForegroundColor Cyan
Write-Host "Backend Port:       $Port" -ForegroundColor Cyan
Write-Host "PocketDB URL:       http://localhost:8090" -ForegroundColor Cyan
Write-Host "Frontend Dev:       http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Yellow
Write-Host "  • Backend API:     http://localhost:$Port" -ForegroundColor Cyan
Write-Host "  • API Docs:        http://localhost:$Port/docs" -ForegroundColor Cyan
Write-Host "  • PocketDB Admin:  http://localhost:8090/_/" -ForegroundColor Cyan
Write-Host "  • Frontend:        http://localhost:5173" -ForegroundColor Cyan
Write-Host ""

# Initialize PocketDB collections (if not disabled)
if (-not $NoPocketDB) {
    Write-Section "Initializing PocketDB Collections"
    try {
        python -c @"
import asyncio
import sys
sys.path.insert(0, 'AGRISENSEFULL-STACK')

async def init():
    try:
        from agrisense_app.backend.database import init_database
        db = await init_database('pocketdb')
        stats = await db.get_stats()
        print('✓ Database initialized successfully')
        await db.close()
    except Exception as e:
        print(f'⚠ Warning: Could not verify PocketDB: {e}')
        print('Make sure PocketDB is running at http://localhost:8090')
        sys.exit(1)

asyncio.run(init())
"@
    } catch {
        Write-Warning "Could not fully initialize PocketDB"
        Write-Host "Make sure PocketDB is running: docker run -d -p 8090:8090 ghcr.io/pocketbase/pocketbase:latest" -ForegroundColor Yellow
    }
}

# Start backend if not frontend-only
if (-not $FrontendOnly) {
    Write-Header "Starting Backend Server"
    Write-Section "Backend startup"
    Write-Host "Starting FastAPI on port $Port..." -ForegroundColor Yellow
    Write-Host "Press Ctrl+C in this window to stop the backend" -ForegroundColor Yellow
    Write-Host ""
    
    # Start backend in same window (blocking)
    $backendCmd = "uvicorn AGRISENSEFULL-STACK.src.backend.main:app --host 127.0.0.1 --port $Port --reload"
    Write-Host "Running: $backendCmd" -ForegroundColor DarkGray
    Write-Host ""
    
    & cmd /c "cd AGRISENSEFULL-STACK && $backendCmd"
}

Write-Host ""
Write-Success "Server stopped"
