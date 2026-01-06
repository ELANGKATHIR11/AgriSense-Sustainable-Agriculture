"""
Simple model fetcher used at container start.
Behavior:
- If destination directory is non-empty, do nothing.
- If empty and environment variable MODEL_BASE_URL is set and a manifest exists at
  $MODEL_BASE_URL/model_manifest.json, download listed files into the destination.
- This is intentionally conservative and requires proper URLs/credentials in env vars.

Usage:
  python scripts/fetch_models.py --dest /app/backend/ml_models

Environment variables:
  MODEL_BASE_URL - base URL where models are hosted (required to auto-download)
  MODEL_MANIFEST - optional relative path to manifest (default: model_manifest.json)
  MODEL_AUTH_HEADER - optional auth header value (e.g. "Bearer <token>")

Manifest format (JSON):
  ["model1.pt", "subdir/model2.onnx", ...]

This script is intentionally minimal; adapt to your storage provider (S3, Azure Blob, etc.)
"""
import os
import sys
import json
import argparse
from pathlib import Path

try:
    import requests
except Exception:
    requests = None


def download_file(url, dest, headers=None):
    if requests is None:
        raise RuntimeError('requests not available in environment')
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dest', required=True)
    args = p.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # If dest has files, assume models already present
    if any(dest.rglob('*')):
        print(f"Models directory '{dest}' not empty — skipping fetch.")
        return 0

    base = os.environ.get('MODEL_BASE_URL')
    if not base:
        print('MODEL_BASE_URL not set — skipping model fetch (expected for local dev).')
        return 0

    manifest_name = os.environ.get('MODEL_MANIFEST', 'model_manifest.json')
    manifest_url = base.rstrip('/') + '/' + manifest_name.lstrip('/')
    headers = None
    auth_header = os.environ.get('MODEL_AUTH_HEADER')
    if auth_header:
        headers = {'Authorization': auth_header}

    print(f'Downloading model manifest from {manifest_url} ...')
    if requests is None:
        print('requests library not available; cannot download models. Install requests or include it in your image requirements.')
        return 1

    r = requests.get(manifest_url, headers=headers, timeout=30)
    r.raise_for_status()
    try:
        manifest = r.json()
    except Exception as e:
        print('Failed to parse manifest JSON:', e)
        return 1

    if not isinstance(manifest, list):
        print('Manifest JSON must be a list of relative file paths')
        return 1

    for rel in manifest:
        url = base.rstrip('/') + '/' + rel.lstrip('/')
        dest_path = dest / rel
        print('Downloading', rel, '->', dest_path)
        try:
            download_file(url, dest_path, headers=headers)
        except Exception as e:
            print('Failed to download', url, e)
            return 2

    print('Models downloaded successfully')
    return 0


if __name__ == '__main__':
    sys.exit(main())
