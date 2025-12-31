#!/usr/bin/env bash
set -euo pipefail

# cleanup_prepare.sh
# Dry-run preview and optional apply script for cleanup. Safe by default.

# Configure the candidate paths (relative to repo root)
CANDIDATES=(
  ".venv"
  ".venv-py312"
  ".venv_py312_npu"
  "venv_npu"
  ".venv-ml"
  "node_modules"
  "agrisense_app/frontend/farm-fortune-frontend-main/node_modules"
  "agrisense_app/frontend/farm-fortune-frontend-main/dist"
  "hf-space-temp"
  "AI_Models"
  "ml_models"
  "__pycache__"
)

# Set DRY_RUN=1 (default) to only preview removals. Set CONFIRM=1 to actually apply.
DRY_RUN=${DRY_RUN:-1}
CONFIRM=${CONFIRM:-0}
ARCHIVE_DIR=${ARCHIVE_DIR:-"backup"}
BRANCH_NAME="cleanup/auto-prune"
PATCH_FILE="cleanup/cleanup.patch"

echo "Cleanup script — DRY_RUN=${DRY_RUN}  CONFIRM=${CONFIRM}"

echo "\nPreviewing git removals (dry-run):"
for p in "${CANDIDATES[@]}"; do
  if [ -e "$p" ]; then
    echo "\n==> Candidate: $p"
    git rm --ignore-unmatch --recursive --dry-run "$p" || true
  else
    echo "\n==> Candidate: $p — (missing)"
  fi
done

if [ "$DRY_RUN" -eq 1 ] && [ "$CONFIRM" -ne 1 ]; then
  echo "\nDry-run complete. To apply, set DRY_RUN=0 and CONFIRM=1 and re-run this script."
  echo "Example: DRY_RUN=0 CONFIRM=1 ARCHIVE_DIR=../backup bash cleanup/cleanup_prepare.sh"
  exit 0
fi

# Confirm apply
if [ "$CONFIRM" -ne 1 ]; then
  echo "CONFIRM not set. Exiting without making changes."
  exit 0
fi

# Create archive dir
mkdir -p "$ARCHIVE_DIR"
TS=$(date +%Y%m%d_%H%M%S)
ARCHIVE="$ARCHIVE_DIR/cleanup-archive-${TS}.tar.gz"

echo "Archiving selected candidates to $ARCHIVE (this may take a while)"
# Only archive the ones that exist
TO_ARCHIVE=()
for p in "${CANDIDATES[@]}"; do
  if [ -e "$p" ]; then
    TO_ARCHIVE+=("$p")
  fi
done

if [ ${#TO_ARCHIVE[@]} -gt 0 ]; then
  tar -czf "$ARCHIVE" "${TO_ARCHIVE[@]}"
  echo "Archive created: $ARCHIVE"
else
  echo "No files to archive."
fi

# Create branch
echo "Creating branch $BRANCH_NAME"
git checkout -b "$BRANCH_NAME"

# Remove from git and commit
for p in "${CANDIDATES[@]}"; do
  echo "Removing (if tracked): $p"
  git rm --ignore-unmatch --recursive "$p" || true
done

COMMIT_MSG="chore: cleanup large artifacts and environments (dry-run applied)"
git commit -m "$COMMIT_MSG" || echo "No changes to commit"

# Write patch
git diff HEAD~1 HEAD > "$PATCH_FILE" || echo "No patch generated"

echo "Cleanup commit created on branch $BRANCH_NAME"
echo "Patch written to $PATCH_FILE"

echo "Done. Run tests and review changes. To push: git push origin $BRANCH_NAME"
