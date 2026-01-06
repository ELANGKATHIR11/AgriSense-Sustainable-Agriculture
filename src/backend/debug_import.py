#!/usr/bin/env python
"""Debug script to capture full import traceback"""
import sys
import traceback

print("=" * 70)
print("Attempting to import main module...")
print("=" * 70)

try:
    import main
    print("\n✓ Successfully imported main")
except Exception as e:
    print(f"\n✗ Import failed with {type(e).__name__}:")
    print(f"\nFull traceback:")
    traceback.print_exc()
    print("\n" + "=" * 70)
    print("ERROR SUMMARY")
    print("=" * 70)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    sys.exit(1)
