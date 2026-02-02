import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent

# Run flake8 and capture F401 unused-import warnings
try:
    out = subprocess.check_output(
        ["flake8", "backend/ml", "--max-line-length=79"],
        stderr=subprocess.STDOUT,
    ).decode("utf-8", errors="ignore")
except subprocess.CalledProcessError as e:
    out = e.output.decode("utf-8", errors="ignore")

pattern = re.compile(
    r"^(?P<path>[^:]+):\d+:\d+:\s+F401\s+'(?P<imp>[^']+)'", re.M
)
matches = pattern.findall(out)

if not matches:
    print("No F401 matches found.")
else:
    by_file = {}
    for path, imp in matches:
        by_file.setdefault(path, set()).add(imp)

    for path, imps in by_file.items():
        p = Path(path)
        if not p.exists():
            print(f"File not found: {p}")
            continue
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        new_lines = []
        removed_any = False
        for ln in lines:
            stripped = ln.strip()
            removed = False
            for imp in imps:
                # imp may be "numpy as np" or "os" etc.
                base = imp.split()[0]
                # Match top-level import lines that reference the base name
                if re.match(rf"\s*import\s+.*\b{re.escape(base)}\b", ln):
                    removed = True
                    break
                if re.match(
                    rf"\s*from\s+.*\b{re.escape(base)}\b\s+import\s+.+", ln
                ):
                    removed = True
                    break
            if removed:
                removed_any = True
                continue
            new_lines.append(ln)
        if removed_any:
            new_text = "\n".join(new_lines).rstrip() + "\n"
            p.write_text(new_text, encoding="utf-8")
            print(f"Removed unused imports in {p}")
