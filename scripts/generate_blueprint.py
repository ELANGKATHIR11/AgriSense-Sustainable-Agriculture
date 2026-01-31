#!/usr/bin/env python3
"""
Simple blueprint generator stub.
- Reads PROJECT_BLUEPRINT.md and writes AGRISENSE_BLUEPRINT.md with a timestamp.
- Designed to be lightweight and safe for CI (sets AGRISENSE_DISABLE_ML to avoid heavy imports).
"""
import datetime
import os
import sys

INPUT = 'PROJECT_BLUEPRINT.md'
OUTPUT = 'AGRISENSE_BLUEPRINT.md'


def main():
    if os.environ.get('AGRISENSE_DISABLE_ML') == '1':
        # Ensure script doesn't import heavy ML libs in CI
        pass

    if not os.path.exists(INPUT):
        print(f"Input {INPUT} not found. No blueprint generated.")
        sys.exit(0)

    with open(INPUT, 'r', encoding='utf-8') as fh:
        content = fh.read()

    header = f"# AGRISENSE Blueprint (generated)\n# Generated: {datetime.datetime.utcnow().isoformat()}Z\n\n"
    out = header + content

    with open(OUTPUT, 'w', encoding='utf-8') as fh:
        fh.write(out)

    print(f"Wrote {OUTPUT}")


if __name__ == '__main__':
    main()
