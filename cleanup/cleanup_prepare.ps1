# cleanup_prepare.ps1
# PowerShell equivalent of cleanup/cleanup_prepare.sh
param()

Set-StrictMode -Version Latest

function Write-Status($msg, $level='INFO') {
    Write-Host "[$level] $msg"
}

# Candidates to remove
$candidates = @(
    '.venv',
    '.venv-py312',
    '.venv_py312_npu',
    'venv_npu',
    '.venv-ml',
    'node_modules',
    'agrisense_app/frontend/farm-fortune-frontend-main/node_modules',
    'agrisense_app/frontend/farm-fortune-frontend-main/dist',
    'hf-space-temp',
    '__pycache__',
    '.pytest_cache',
    'smoke-output',
    'smoke',
    'deployed',
    'temp_model.onnx.data',
    'training_output.log',
    'gpu_training_20251228_110532.log'
)

# Read env vars
$DRY_RUN = if ($env:DRY_RUN) { [int]$env:DRY_RUN } else { 1 }
$CONFIRM = if ($env:CONFIRM) { [int]$env:CONFIRM } else { 0 }
$ARCHIVE_DIR = if ($env:ARCHIVE_DIR) { $env:ARCHIVE_DIR } else { '..\backup' }
$BRANCH_NAME = 'cleanup/auto-prune'
$PATCH_FILE = 'cleanup\cleanup.patch'

Write-Status "Cleanup script — DRY_RUN=$DRY_RUN  CONFIRM=$CONFIRM"

Write-Status "\nPreviewing git removals (dry-run):"
# Detect active virtualenv and avoid deleting it accidentally
$activeVenv = $null
try {
    if ($env:VIRTUAL_ENV) { $activeVenv = (Resolve-Path $env:VIRTUAL_ENV).ProviderPath }
    else {
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCmd) {
            $pythonPath = $pythonCmd.Source
            # typically ...\\<venv>\\Scripts\\python.exe — remove \\Scripts\\python.exe
            $activeVenv = Split-Path (Split-Path $pythonPath -Parent) -Parent
        }
    }
} catch {
    $activeVenv = $null
}
if ($activeVenv) { Write-Status "Detected active virtualenv: $activeVenv" } else { Write-Status "No active virtualenv detected." }
foreach ($p in $candidates) {
    if (Test-Path $p) {
        $resolved = (Resolve-Path $p).ProviderPath
        if ($activeVenv -and ($resolved -like "$activeVenv*" -or $activeVenv -like "$resolved*")) {
            Write-Status "Candidate: $p — SKIPPED (active virtualenv)"
            continue
        }
        Write-Status "Candidate: $p"
        & git rm --ignore-unmatch --recursive --dry-run -- "$p" 2>&1 | Write-Output
    }
    else {
        Write-Status "Candidate: $p — (missing)"
    }
}

if ($DRY_RUN -eq 1 -and $CONFIRM -ne 1) {
    Write-Status "\nDry-run complete. To apply, set environment variables DRY_RUN=0 and CONFIRM=1 and re-run this script." "INFO"
    exit 0
}

if ($CONFIRM -ne 1) {
    Write-Status "CONFIRM not set. Exiting without making changes." "WARNING"
    exit 0
}

# Ensure archive dir exists
if (-not (Test-Path $ARCHIVE_DIR)) { New-Item -ItemType Directory -Path $ARCHIVE_DIR | Out-Null }
$TS = Get-Date -Format 'yyyyMMdd_HHmmss'
$ARCHIVE = Join-Path $ARCHIVE_DIR "cleanup-archive-$TS.zip"

Write-Status "Archiving selected candidates to $ARCHIVE (this may take a while)"
$toArchive = @()
foreach ($p in $candidates) {
    if (Test-Path $p) { $toArchive += $p }
}

# Find repository .bak files (likely duplicates/backups) but skip .git and major venv folders
Write-Status "Searching for '*.bak' files to remove/archive"
$bakFiles = Get-ChildItem -Path $PSScriptRoot -Recurse -Include *.bak -File -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\.git\\' -and $_.FullName -notmatch '\\.venv' -and $_.FullName -notmatch '\\venv_npu' -and $_.FullName -notmatch '\\.venv-py312' }
if ($bakFiles.Count -gt 0) {
    Write-Status "Found $($bakFiles.Count) .bak files. Adding to archive and removal list."
    foreach ($bf in $bakFiles) { $toArchive += $bf.FullName }
}

if ($toArchive.Count -gt 0) {
    try {
        Compress-Archive -Path $toArchive -DestinationPath $ARCHIVE -Force
        Write-Status "Archive created: $ARCHIVE" "SUCCESS"
    }
    catch {
        Write-Status "Archive failed: $_" "ERROR"
        exit 1
    }
}
else {
    Write-Status "No files to archive." "INFO"
}

# Create git branch
Write-Status "Creating branch $BRANCH_NAME"
$checkout = & git checkout -b $BRANCH_NAME 2>&1
Write-Output $checkout

# Remove from git and commit
foreach ($p in $candidates) {
    Write-Status "Removing (if tracked): $p"
    & git rm --ignore-unmatch --recursive -- "$p" 2>&1 | Write-Output
}

# Also remove discovered .bak files
if ($bakFiles -and $bakFiles.Count -gt 0) {
    foreach ($bf in $bakFiles) {
        Write-Status "Removing bak file: $($bf.FullName)"
        & git rm --ignore-unmatch -- "$($bf.FullName)" 2>&1 | Write-Output
        try { Remove-Item -Path $bf.FullName -Force -ErrorAction SilentlyContinue } catch { }
    }
}

$commitMsg = 'chore: cleanup large artifacts and environments (applied via cleanup_prepare.ps1)'
$commit = & git commit -m $commitMsg 2>&1
Write-Output $commit

# Write patch
try {
    $diff = & git diff HEAD~1 HEAD
    if ($diff) {
        $diff | Out-File -FilePath $PATCH_FILE -Encoding utf8
        Write-Status "Patch written to $PATCH_FILE" "SUCCESS"
    }
    else {
        Write-Status "No diff produced; no patch written." "INFO"
    }
}
catch {
    Write-Status "Failed to write patch: $_" "ERROR"
}

Write-Status "Cleanup commit created on branch $BRANCH_NAME"
Write-Status "Done. Run tests and review changes. To push: git push origin $BRANCH_NAME"