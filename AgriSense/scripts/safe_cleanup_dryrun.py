import os
from pathlib import Path
import fnmatch

root = Path('.').resolve()
# Patterns for directories to remove
dir_patterns = [
    '__pycache__', '.pytest_cache', '.mypy_cache', 'node_modules', 'build', 'dist',
    '.next', 'coverage', '.venv', 'venv', 'env', '.env', 'logs', 'tmp', 'temp', 'output'
]
# File patterns (extensions) to remove
file_patterns = [
    '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.exe',
    '*.ckpt', '*.pt', '*.pth', '*.onnx', '*.h5', '*.pb',
    '*.zip', '*.tar.gz', '*.tar', '*.tgz', '*.gz', '*.tar.bz2',
    '*.log', '*.sqlite', '*.db', '*.sqlite3',
]
# Model folders we want to flag
model_dirs = ['ml_models', 'models', 'checkpoints', 'checkpoints_new']

matches = []
total = 0

for dirpath, dirnames, filenames in os.walk(root):
    pdir = Path(dirpath)
    # check dir patterns
    for d in dir_patterns + model_dirs:
        if d in pdir.parts:
            try:
                # calculate size of this folder (only once)
                size = 0
                for p2, _, files2 in os.walk(pdir):
                    for f2 in files2:
                        try:
                            fp = Path(p2) / f2
                            if fp.is_file():
                                size += fp.stat().st_size
                        except Exception:
                            pass
                matches.append((str(pdir), 'dir', size))
                total += size
                # prune walking into this dir by clearing dirnames if exact match
                # but still allow other matches
                break
            except Exception:
                pass
    # check files
    for fn in filenames:
        for pat in file_patterns:
            if fnmatch.fnmatch(fn, pat):
                fp = pdir / fn
                try:
                    size = fp.stat().st_size
                except Exception:
                    size = 0
                matches.append((str(fp), 'file', size))
                total += size
                break

# Deduplicate by path
seen = set()
uniq = []
for path, typ, size in matches:
    if path not in seen:
        seen.add(path)
        uniq.append((path, typ, size))

uniq_sorted = sorted(uniq, key=lambda x: x[2], reverse=True)

print(f"Found {len(uniq_sorted)} candidate items (dry-run).")
print()
for path, typ, size in uniq_sorted[:200]:
    # human readable
    hr = size
    unit = 'B'
    for u in ['B','KB','MB','GB','TB']:
        if hr < 1024:
            break
        hr = hr / 1024.0
        unit = {'B':'KB','KB':'MB','MB':'GB','GB':'TB'}.get(unit, unit)
    if unit == 'B':
        hr_str = f"{hr:.0f} {unit}"
    else:
        hr_str = f"{hr:.2f} {unit}"
    print(f"{typ.upper():4} {hr_str:10}  {path}")

# summary
hr = total
unit = 'B'
for u in ['B','KB','MB','GB','TB']:
    if hr < 1024:
        break
    hr = hr / 1024.0
    unit = {'B':'KB','KB':'MB','MB':'GB','GB':'TB'}.get(unit, unit)
if unit == 'B':
    hr_str = f"{hr:.0f} {unit}"
else:
    hr_str = f"{hr:.2f} {unit}"
print()
print(f"Total reclaimable (dry-run): {hr_str} ({total} bytes)")
print()
print("Notes: This is a conservative dry-run. Review listed paths before deleting. To actually delete, run the appropriate cleanup script with --apply or handle manually.")
