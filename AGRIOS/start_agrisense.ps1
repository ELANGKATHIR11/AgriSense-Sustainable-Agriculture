Write-Host "Starting AgriSense System..." -ForegroundColor Green

$root = "f:\Agrisense-A samart Agriculture Solution\Agrisense\AgriSense"
$backendPath = Join-Path $root "agrisense_app\backend"
$frontendPath = Join-Path $root "agrisense_app\frontend\farm-fortune-frontend-main"

# Start Backend
Write-Host "Launching Backend in new window..." -ForegroundColor Cyan
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; Write-Host 'Starting Backend...'; if (Test-Path .venv) { .\.venv\Scripts\Activate.ps1 }; uvicorn main:app --reload --host 0.0.0.0 --port 8000"

# Start Frontend
Write-Host "Launching Frontend in new window..." -ForegroundColor Cyan
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; Write-Host 'Starting Frontend...'; npm run dev"

Write-Host "All services launched! Check the new windows for logs." -ForegroundColor Green
