import hashlib
from pathlib import Path

lock = Path('agrisense_app/backend/core/CORE_HASH.lock')
core_dir = Path('agrisense_app/backend/core')
if not lock.exists():
    print('CORE_HASH.lock not found')
    raise SystemExit(2)

entries = []
for line in lock.read_text().splitlines():
    if not line.strip():
        continue
    parts = line.split(None,1)
    if len(parts)!=2:
        continue
    h, rel = parts
    entries.append((rel.strip(), h.strip()))

ok = True
for rel, expected in entries:
    fp = core_dir / rel
    if rel == 'CORE_HASH.lock':
        continue
    if not fp.exists():
        print('MISSING:', rel)
        ok = False
        continue
    data = fp.read_bytes()
    h = hashlib.sha256(data).hexdigest()
    if h!=expected:
        print('MISMATCH:', rel)
        ok = False

if ok:
    print('All core files match CORE_HASH.lock')
    raise SystemExit(0)
else:
    print('Core verification failed')
    raise SystemExit(1)
