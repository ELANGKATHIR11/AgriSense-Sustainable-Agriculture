import re
from pathlib import Path

root = Path(__file__).resolve().parent
py_files = list(root.rglob("*.py"))

fstring_re = re.compile(r"(?P<prefix>\b)f(?P<quote>['\"])")

for path in py_files:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    new_lines = []
    changed = False
    for ln in lines:
        # Detect simple single-line f-strings without placeholders
        m = re.search(r"\bf(['\"]).*\1", ln)
        if m:
            # extract the string content
            quote = m.group(1)
            try:
                s = ln.split(quote, 1)[1]
                content = s.rsplit(quote, 1)[0]
            except Exception:
                content = ""
            if "{" not in content and "}" not in content:
                # remove the leading 'f' before the quote
                new_ln = ln.replace("f" + quote, quote, 1)
                if new_ln != ln:
                    ln = new_ln
                    changed = True
        new_lines.append(ln)
    new_text = "\n".join(new_lines) + "\n"
    if changed and new_text != text:
        path.write_text(new_text, encoding="utf-8")
        print(f"Fixed f-strings: {path}")
