from pathlib import Path

root = Path(__file__).resolve().parent
py_files = list(root.rglob("*.py"))

for path in py_files:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        continue
    # Normalize line endings and split
    lines = text.splitlines()
    new_lines = []
    changed = False
    for ln in lines:
        if ln.strip() == "":
            if ln != "":
                changed = True
            new_lines.append("")
        else:
            stripped = ln.rstrip()
            if stripped != ln:
                changed = True
            new_lines.append(stripped)
    new_text = "\n".join(new_lines).rstrip() + "\n"
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        print(f"Cleaned: {path}")
