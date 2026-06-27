#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%3.1f %s%s" % (num, 'E', suffix)

candidates = [
    '.venv', '.venv-ml', '.venv-py312', '.venv_py312_npu', 'venv_npu',
    'node_modules',
    'agrisense_app/frontend/farm-fortune-frontend-main/node_modules',
    'agrisense_app/frontend/farm-fortune-frontend-main/dist',
    'AI_Models', 'models', 'ml_models', 'hf-space-temp', 'deployed',
    'smoke-output', '__pycache__', '.pytest_cache', 'smoke',
    'training_output.log', 'gpu_training_20251228_110532.log', 'temp_model.onnx.data'
]

root = Path('.')
results = []

for rel in candidates:
    p = root / rel
    if not p.exists():
        results.append((rel, False, 0, 0, []))
        continue
    total = 0
    files = []
    if p.is_file():
        try:
            total = p.stat().st_size
            files.append((p, total))
        except Exception:
            pass
    else:
        for dirpath, dirnames, filenames in os.walk(p):
            for fn in filenames:
                fp = Path(dirpath) / fn
                try:
                    sz = fp.stat().st_size
                    total += sz
                    files.append((fp, sz))
                except Exception:
                    pass
    files.sort(key=lambda x: x[1], reverse=True)
    results.append((rel, True, total, len(files), files[:20]))

# Print a human-readable summary
print("CLEANUP DRY-RUN REPORT\n")
for rel, exists, total, count, tops in results:
    if not exists:
        print(f"[MISSING] {rel}")
        continue
    print(f"{rel}: {sizeof_fmt(total)} in {count} files")
    for fp, sz in tops:
        relp = fp.relative_to(root)
        print(f"  - {relp} — {sizeof_fmt(sz)}")
    print()

# Also print an overall top 40 largest files in repo
all_files = []
for dirpath, dirnames, filenames in os.walk('.'):
    for fn in filenames:
        fp = Path(dirpath) / fn
        try:
            all_files.append((fp, fp.stat().st_size))
        except Exception:
            pass
all_files.sort(key=lambda x: x[1], reverse=True)
print('TOP 40 LARGEST FILES IN REPO:')
for fp, sz in all_files[:40]:
    print(f"{fp.relative_to(root)}|{sizeof_fmt(sz)}")

# Exit with success
sys.exit(0)
