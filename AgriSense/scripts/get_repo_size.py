import os
from pathlib import Path

def human_readable(n):
    units = ['B','KB','MB','GB','TB']
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024
        i += 1
    return f, units[i]

root = Path('.')
total = 0
for dirpath, dirnames, filenames in os.walk(root):
    for fn in filenames:
        try:
            fp = Path(dirpath) / fn
            if fp.is_file():
                total += fp.stat().st_size
        except Exception:
            pass

f, unit = human_readable(total)
print(f"{f:.2f} {unit}")
print(total)
