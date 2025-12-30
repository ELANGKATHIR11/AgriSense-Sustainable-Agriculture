# Cleanup Deletion Plan (Dry-Run)

This document describes the recommended cleanup and provides safe scripts to preview and apply deletions.

Recommended candidate groups (based on dry-run scan):
- Virtual environments (recreate from requirements):
  - `.venv` (8.6 GB)
  - `.venv-py312` (2.8 GB)
  - `.venv_py312_npu` (2.9 GB)
  - `venv_npu` (4.6 GB)
  - `.venv-ml` (small)

- Frontend dependencies and build artifacts:
  - `agrisense_app/frontend/farm-fortune-frontend-main/node_modules` (590 MB)
  - top-level `node_modules` (36 MB)
  - `agrisense_app/frontend/farm-fortune-frontend-main/dist` (2.4 MB) — keep if you want static asset serving; otherwise safe to remove

- Temporary / deployment artifacts:
  - `hf-space-temp` (636 MB)
  - `deployed/` (empty placeholder)

- Model backups and large data (archive before deletion recommended):
  - `AI_Models` (1.5 GB)
  - `ml_models` (806 MB)

- Other small items: `__pycache__`, `.pytest_cache`, `smoke-output`, log files

Safety & workflow
1. Dry-run: the provided script (`cleanup/cleanup_prepare.sh`) will run git dry-run commands to preview which tracked files would be removed.
2. Archive (recommended): before deleting model/data or environments, create a compressed archive stored outside the repo or in `backup/`.
3. Create branch: the script can create a branch `cleanup/auto-prune` and stage the deletions there, so nothing is committed to `main` directly.
4. Tests: after staging the deletions, run test suite (`pytest`) and a quick lint/type check. If anything breaks, revert the branch.
5. Apply: if everything is OK, merge the cleanup branch.

Commands summary (preview only):
- Preview: `bash cleanup/cleanup_prepare.sh` (will show git dry-run output)
- Apply: `DRY_RUN=0 CONFIRM=1 bash cleanup/cleanup_prepare.sh` (will archive selected dirs, create branch, remove files, commit, and write a patch to `cleanup/cleanup.patch`)

Notes
- The script uses `git rm --ignore-unmatch --recursive` so untracked files are skipped and the script is safe to run multiple times.
- The script will NOT delete files locally until you set `CONFIRM=1` (explicit consent required).
- Archiving large model artifacts requires sufficient disk space; you can set `ARCHIVE_DIR` env var to control destination.
