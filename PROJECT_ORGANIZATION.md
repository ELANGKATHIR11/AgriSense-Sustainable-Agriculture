# Project Organization and Quick Start

This file explains the repository layout, quick setup commands, and common developer workflows to make the project easy to navigate.

Top-level layout (important folders):

- `agrisense_app/` — backend FastAPI app, data, models and services.
- `frontend/` — React + Vite frontend (if present).
- `tools/` — helper scripts (data generation, NPU tools, model conversion).
- `docs/` — documentation and design guides.
- `cleanup/` — cleanup helpers and deletion plans.
- `deployed/` — generated deployment artifacts (ignored in VCS if large).

Key files and commands:

- Setup (PowerShell): `scripts/setup_repo.ps1` — create venv, install deps, enable Git LFS.
- Start backend (dev): `start_agrisense.ps1` or `python start_agrisense.py`.
- Build frontend: `npm install` then `npm run build` inside the frontend folder.
- Tests: `pytest -q` (run inside a virtualenv with dev deps installed).

Data & Models:
- Small datasets: `agrisense_app/backend/data/` (CSV, JSONL manifests)
- Trained models and large artifacts: `agrisense_app/backend/models/` (don't commit large binaries)
- Local model cache was previously stored in `ml_models/` — archived to `backup_ml_models.zip` (moved outside repo). Use the backup or remote storage to restore.

Repository hygiene and recommendations:

- Never commit virtual environments or large model binaries. Use `.gitignore` (already updated) and Git LFS for large files.
- Use `scripts/setup_repo.ps1` on Windows to create a reproducible local dev environment.
- For CI/CD, prefer containerized builds (Dockerfile.* already present).

Quick commands (PowerShell):
```powershell
# create venv and install
.\scripts\setup_repo.ps1

# run backend (example)
& .\venv\Scripts\Activate.ps1
python start_agrisense.py
```

If you want, I can also generate an interactive `README.md` summary or add a `Makefile`/task runner for common developer tasks.
