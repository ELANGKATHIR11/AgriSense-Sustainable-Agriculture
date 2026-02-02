# AgriSense Project Cleanup Script
# Removes build artifacts and dependencies to reduce file count and path length for zipping

Write-Host "🧹 Starting cleanup for 'AgriSense'..." -ForegroundColor Cyan

# 1. Clean Backend
if (Test-Path "backend\node_modules") {
    Write-Host "Removing backend\node_modules..." -ForegroundColor Yellow
    Remove-Item -Path "backend\node_modules" -Recurse -Force -ErrorAction SilentlyContinue
}

# 2. Clean Frontend
if (Test-Path "frontend\node_modules") {
    Write-Host "Removing frontend\node_modules..." -ForegroundColor Yellow
    Remove-Item -Path "frontend\node_modules" -Recurse -Force -ErrorAction SilentlyContinue
}
if (Test-Path "frontend\dist") {
    Write-Host "Removing frontend\dist..." -ForegroundColor Yellow
    Remove-Item -Path "frontend\dist" -Recurse -Force -ErrorAction SilentlyContinue
}

# 3. Clean Python Virtual Environment
if (Test-Path ".venv") {
    Write-Host "Removing .venv..." -ForegroundColor Yellow
    Remove-Item -Path ".venv" -Recurse -Force -ErrorAction SilentlyContinue
}

# 4. Clean Nested Redundant Folder (Ask user manually, but here we just list it)
if (Test-Path "AGRISENSEFULL-STACK\AGRISENSEFULL-STACK") {
    Write-Host "⚠️  Found nested 'AGRISENSEFULL-STACK' folder. Consider deleting it manually if it's a duplicate." -ForegroundColor Magenta
}

Write-Host "`n✅ Cleanup Complete!" -ForegroundColor Green
Write-Host "You can now zip the 'AGRISENSEFULL-STACK' folder easily." -ForegroundColor Green
Write-Host "To reinstall dependencies later:" -ForegroundColor White
Write-Host "  1. Backend: cd backend; npm install" 
Write-Host "  2. Frontend: cd frontend; npm install" 
Write-Host "  3. Python: python -m venv .venv; .venv\Scripts\activate; pip install -r backend\requirements.txt" 
