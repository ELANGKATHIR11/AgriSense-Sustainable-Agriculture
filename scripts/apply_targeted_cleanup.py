import os
import shutil
from pathlib import Path

root = Path('.').resolve()
remove_names_prefix = ('venv', '.venv')
remove_exact = ('node_modules',)

removed = []
reclaimed = 0

for dirpath, dirnames, filenames in os.walk(root, topdown=True):
    # copy list to avoid modification during iteration
    dirs = list(dirnames)
    for d in dirs:
        full = Path(dirpath) / d
        name = d
        # Decide if this dir should be removed
        if name in remove_exact or name.startswith(remove_names_prefix):
            # skip .git and .github accidentally
            if full.match('**/.git') or full.match('**/.github'):
                continue
            try:
                # compute size
                size = 0
                for p, _, files in os.walk(full):
                    for f in files:
                        try:
                            fp = Path(p) / f
                            if fp.is_file():
                                size += fp.stat().st_size
                        except Exception:
                            pass
                # remove
                shutil.rmtree(full)
                removed.append((str(full), size))
                reclaimed += size
                # remove from dirnames to avoid walking into it
                dirnames.remove(d)
            except Exception as e:
                print(f"Failed to remove {full}: {e}")

# print summary
print(f"Removed {len(removed)} directories.")
for path, size in removed:
    hr = size
    unit = 'B'
    for u in ['B','KB','MB','GB','TB']:
        if hr < 1024:
            break
        hr = hr / 1024.0
        unit = {'B':'KB','KB':'MB','MB':'GB','GB':'TB'}.get(unit, unit)
    hr_str = f"{hr:.2f} {unit}" if unit!='B' else f"{hr:.0f} {unit}"
    print(f"- {path}  ({hr_str})")

hr = reclaimed
unit = 'B'
for u in ['B','KB','MB','GB','TB']:
    if hr < 1024:
        break
    hr = hr / 1024.0
    unit = {'B':'KB','KB':'MB','MB':'GB','GB':'TB'}.get(unit, unit)
hr_str = f"{hr:.2f} {unit}" if unit!='B' else f"{hr:.0f} {unit}"
print(f"Total reclaimed: {hr_str} ({reclaimed} bytes)")
