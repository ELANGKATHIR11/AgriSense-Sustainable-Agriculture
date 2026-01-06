# AgriSense Backend ngrok Setup Script

Write-Host "`n=== AgriSense Backend ngrok Tunnel ===" -ForegroundColor Cyan
Write-Host "`nStep 1: Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken`n"

$TOKEN = Read-Host "Paste your ngrok authtoken"

if([string]::IsNullOrWhiteSpace($TOKEN)) {
    Write-Host "Error: Authtoken is required" -ForegroundColor Red
    exit 1
}

$ngrokPath = "$env:TEMP\ngrok.exe"

if(-not (Test-Path $ngrokPath)) {
    Write-Host "ngrok not found. Downloading..." -ForegroundColor Yellow
    $zipPath = "$env:TEMP\ngrok.zip"
    Invoke-WebRequest -Uri "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip" -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $env:TEMP -Force
}

Write-Host "`nInstalling authtoken..." -ForegroundColor Yellow
& $ngrokPath config add-authtoken $TOKEN

Write-Host "`nStarting ngrok tunnel to localhost:8004..." -ForegroundColor Green
Write-Host "The public URL will appear below - copy it to use with the frontend`n"

& $ngrokPath http 8004
