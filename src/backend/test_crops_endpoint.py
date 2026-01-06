#!/usr/bin/env python
"""Test the backend crops endpoint"""
import urllib.request
import json
import sys

url = "http://127.0.0.1:8004/api/vlm/crops"

print(f"\n{'='*70}")
print(f"Testing: {url}")
print('='*70)

try:
    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.loads(response.read().decode())
        print(f"Status: {response.status}")
        print(f"\nResponse type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            if 'items' in data:
                print(f"Number of items: {len(data['items'])}")
                if data['items']:
                    print(f"\nFirst crop:")
                    print(json.dumps(data['items'][0], indent=2))
        elif isinstance(data, list):
            print(f"Number of items: {len(data)}")
            if data:
                print(f"\nFirst crop:")
                print(json.dumps(data[0], indent=2))
        print(f"\nFull response (first 1500 chars):")
        print(json.dumps(data, indent=2)[:1500])
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
