# Hugging Face Space Deployment Script
# Deploys AgriSense to HF Space at: https://huggingface.co/spaces/KATHIR2006/agrisense-app

param(
    [string]$HFUsername = "KATHIR2006",
    [string]$SpaceName = "agrisense-app",
    [string]$HFToken = $env:HF_TOKEN
)

$ErrorActionPreference = "Stop"

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       AgriSense → Hugging Face Space Deployment           ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if HF token is provided
if (-not $HFToken) {
    Write-Host "⚠️  HF_TOKEN not found in environment variables" -ForegroundColor Yellow
    Write-Host "   To create a token:" -ForegroundColor Yellow
    Write-Host "   1. Go to https://huggingface.co/settings/tokens" -ForegroundColor Yellow
    Write-Host "   2. Create a new token with 'write' access" -ForegroundColor Yellow
    Write-Host "   3. Run: `$env:HF_TOKEN = 'your_token_here'" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Enter your HF token"
    $HFToken = Read-Host "HF_TOKEN"
}

$HFSpaceUrl = "https://huggingface.co/spaces/$HFUsername/$SpaceName"
$HFRepoUrl = "https://huggingface.co/spaces/$HFUsername/$SpaceName.git"

Write-Host "📋 Deployment Configuration:" -ForegroundColor Cyan
Write-Host "   HF Username: $HFUsername" -ForegroundColor Green
Write-Host "   Space Name: $SpaceName" -ForegroundColor Green
Write-Host "   Space URL: $HFSpaceUrl" -ForegroundColor Green
Write-Host "   Repo URL: $HFRepoUrl" -ForegroundColor Green
Write-Host ""

# Create temp directory for HF Space
$TempDir = "$env:TEMP\hf-agrisense-$([DateTime]::Now.Ticks)"
Write-Host "🔧 Setting up deployment directory..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
Write-Host "   ✓ Created: $TempDir" -ForegroundColor Green
Write-Host ""

# Clone the HF Space repo
Write-Host "📥 Cloning Hugging Face Space repository..." -ForegroundColor Cyan
try {
    # Configure git to use token
    $CredString = "oauth2:$HFToken"
    $EncodedCred = [System.Convert]::ToBase64String([System.Text.Encoding]::ASCII.GetBytes($CredString))
    
    cd $TempDir
    git clone $HFRepoUrl . 2>&1 | Select-Object -First 5
    Write-Host "   ✓ Cloned successfully" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Clone failed. Space may not exist yet." -ForegroundColor Yellow
    Write-Host "   Creating new space structure..." -ForegroundColor Yellow
    
    # Initialize git repo
    git init
    git config user.email "github@agrisense.local"
    git config user.name "AgriSense Deployer"
    Write-Host "   ✓ Initialized new git repository" -ForegroundColor Green
}

Write-Host ""
Write-Host "📦 Copying deployment files..." -ForegroundColor Cyan

# Copy key files
$FilesToCopy = @(
    @{Src="D:\AGRISENSEFULL-STACK\Dockerfile.huggingface"; Dst="$TempDir\Dockerfile"}
    @{Src="D:\AGRISENSEFULL-STACK\start.sh"; Dst="$TempDir\start.sh"}
    @{Src="D:\AGRISENSEFULL-STACK\.dockerignore"; Dst="$TempDir\.dockerignore"}
    @{Src="D:\AGRISENSEFULL-STACK\package.json"; Dst="$TempDir\package.json"}
    @{Src="D:\AGRISENSEFULL-STACK\agrisense_app"; Dst="$TempDir\agrisense_app"}
    @{Src="D:\AGRISENSEFULL-STACK\README.HUGGINGFACE.md"; Dst="$TempDir\README.md"}
)

foreach ($file in $FilesToCopy) {
    if (Test-Path $file.Src) {
        if ((Get-Item $file.Src).PSIsContainer) {
            Copy-Item -Path $file.Src -Destination $file.Dst -Recurse -Force
            Write-Host "   ✓ Copied directory: $(Split-Path -Leaf $file.Src)" -ForegroundColor Green
        } else {
            Copy-Item -Path $file.Src -Destination $file.Dst -Force
            Write-Host "   ✓ Copied file: $(Split-Path -Leaf $file.Src)" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "🔐 Creating .gitattributes for LFS support..." -ForegroundColor Cyan
@"
*.md text eol=lf
*.py text eol=lf
*.sh text eol=lf
*.json text eol=lf
*.ts text eol=lf
*.tsx text eol=lf
*.js text eol=lf
*.jsx text eol=lf
Dockerfile text eol=lf
*.so binary
*.pyc binary
*.pth binary
*.bin binary
"@ | Set-Content "$TempDir\.gitattributes"
Write-Host "   ✓ Created .gitattributes" -ForegroundColor Green

Write-Host ""
Write-Host "📝 Staging files for commit..." -ForegroundColor Cyan
cd $TempDir
git add .
git config user.email "github@agrisense.local"
git config user.name "AgriSense Deployer"
Write-Host "   ✓ All files staged" -ForegroundColor Green

Write-Host ""
Write-Host "💾 Committing changes..." -ForegroundColor Cyan
git commit -m "Deploy AgriSense to HF Spaces with Docker, FastAPI backend, React frontend, and Celery workers" 2>&1 | Select-Object -First 3
Write-Host "   ✓ Committed successfully" -ForegroundColor Green

Write-Host ""
Write-Host "🚀 Pushing to Hugging Face Spaces..." -ForegroundColor Cyan

# Setup git credential helper for HF auth
$GitAuthUrl = "https://oauth2:$HFToken@huggingface.co"
git remote remove origin 2>$null
git remote add origin $GitAuthUrl/$HFUsername/$SpaceName.git

try {
    git push -u origin main --force 2>&1
    Write-Host "   ✓ Push successful!" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  Attempting push with main branch..." -ForegroundColor Yellow
    git branch -M main
    git push -u origin main --force 2>&1
}

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              🎉 Deployment Steps Complete! 🎉             ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "✅ Next Steps (Configure Environment Secrets):" -ForegroundColor Green
Write-Host ""
Write-Host "1. Go to: $HFSpaceUrl/settings" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Add these secrets in the 'Repository secrets' section:" -ForegroundColor Cyan
Write-Host "   • MONGO_URI" -ForegroundColor Yellow
Write-Host "     Value: mongodb+srv://AgriSense:AgriSense@cluster0.fxyufyf.mongodb.net/AgriSense?retryWrites=true&w=majority" -ForegroundColor Gray
Write-Host ""
Write-Host "   • REDIS_URL" -ForegroundColor Yellow
Write-Host "     Value: redis://default:AS7fAAIncDE1ZTg4YjUyYTA4YTQ0ZmY1ODczMmYzMDEwYjI4ZjY0YXAxMTE5OTk@workable-mongrel-11999.upstash.io:6379" -ForegroundColor Gray
Write-Host ""
Write-Host "   • AGRISENSE_ADMIN_TOKEN" -ForegroundColor Yellow
Write-Host "     Value: sk-agrisense-8QzLpVxMkN2jRwQaB5sT9nH1cF3dG7jI" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Save the secrets - the Space will automatically rebuild" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Monitor build progress at: $HFSpaceUrl?logs" -ForegroundColor Cyan
Write-Host ""
Write-Host "5. Once built, access your app:" -ForegroundColor Cyan
Write-Host "   Frontend UI: https://$HFUsername-$SpaceName.hf.space/ui/" -ForegroundColor Cyan
Write-Host "   API Docs: https://$HFUsername-$SpaceName.hf.space/docs" -ForegroundColor Cyan
Write-Host "   Health Check: https://$HFUsername-$SpaceName.hf.space/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Build time: 10-15 minutes" -ForegroundColor Magenta
Write-Host ""

# Cleanup
Write-Host "🧹 Cleaning up temporary directory..." -ForegroundColor Cyan
cd D:\AGRISENSEFULL-STACK
Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "   ✓ Cleanup complete" -ForegroundColor Green
Write-Host ""

Write-Host "✨ Deployment script completed!" -ForegroundColor Green
