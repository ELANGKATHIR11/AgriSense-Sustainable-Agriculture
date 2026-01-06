#!/usr/bin/env python
"""Test the backend crops endpoint"""
import urllib.request
import json

urls = [
    "http://127.0.0.1:8004/health",
    "http://127.0.0.1:8004/api/vlm/crops",
]

for url in urls:
    print(f"\n{'='*60}")
    print(f"Testing: {url}")
    print('='*60)
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            print(f"Status: {response.status}")
            print(json.dumps(data, indent=2)[:500])
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
