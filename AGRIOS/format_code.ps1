# Auto-format AgriSense Backend Code
# This script will automatically fix code style issues

Write-Host "🔧 AgriSense Code Formatter" -ForegroundColor Green
Write-Host "============================`n" -ForegroundColor Green

$backendPath = "f:\Agrisense-A samart Agriculture Solution\Agrisense\AgriSense\agrisense_app\backend"

# Check if Black is installed
Write-Host "Checking for Black formatter..." -ForegroundColor Cyan
try {
    $blackVersion = black --version 2>&1
    Write-Host "✓ Black is installed: $blackVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Black not found. Installing..." -ForegroundColor Yellow
    pip install black isort autopep8
}

# Confirm with user
Write-Host "`nThis will reformat Python files in:" -ForegroundColor Yellow
Write-Host "  $backendPath" -ForegroundColor White
Write-Host "`nLine length will be set to 120 characters (modern PEP 8 standard)" -ForegroundColor Gray
$confirm = Read-Host "`nContinue? (y/n)"

if ($confirm -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit
}

Write-Host "`n📝 Formatting Python files..." -ForegroundColor Cyan
black $backendPath --line-length 120

Write-Host "`n📦 Sorting imports..." -ForegroundColor Cyan
isort $backendPath --profile black --line-length 120

Write-Host "`n✨ Code formatting complete!" -ForegroundColor Green
Write-Host "`n📊 Checking for remaining issues..." -ForegroundColor Cyan
flake8 "$backendPath\main.py" --count --select=E9,F63,F7,F82 --show-source --statistics

Write-Host "`n✅ Done! Your code is now formatted according to modern Python standards." -ForegroundColor Green
Write-Host "   Review the changes and commit if satisfied." -ForegroundColor Gray
