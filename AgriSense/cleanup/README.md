Cleanup helper files

Files added:
- `DELETION_PLAN.md` — explains candidates, rationale, and process.
- `cleanup_prepare.sh` — dry-run preview and optional apply script.

Usage (preview only):

```bash
# From repo root
bash cleanup/cleanup_prepare.sh
```

To actually apply (archive -> create branch -> remove -> commit -> create patch):

```bash
# Use with caution; this will create archives and modify git index/branch
DRY_RUN=0 CONFIRM=1 ARCHIVE_DIR=../backup bash cleanup/cleanup_prepare.sh
```

If you prefer PowerShell on Windows, run the script in Git Bash or WSL. I can add a PowerShell equivalent on request.
