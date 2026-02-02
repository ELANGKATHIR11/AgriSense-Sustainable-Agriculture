# AgriSense Full-Stack Dev Runner
# Run backend and frontend in separate windows.

$root = $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"

Write-Host "Starting AgriSense Backend (port 5000)..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$backend'; node server.js"

Start-Sleep -Seconds 4

Write-Host "Starting AgriSense Frontend (port 3001)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$frontend'; npm run dev"

Write-Host ""
Write-Host "Backend:  http://localhost:5000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:5000/api/docs" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3001" -ForegroundColor Cyan
Write-Host ""
Write-Host "Close the backend and frontend PowerShell windows to stop."
