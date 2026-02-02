#!/usr/bin/env python3
"""
Setup script to generate all edge AI knowledge bases
Run this to initialize cultivation guides and disease knowledge
"""

import sys
from pathlib import Path


def main():
    print("=" * 80)
    print("EDGE AI KNOWLEDGE BASE SETUP")
    print("=" * 80)
    print()

    # Import generators
    try:
        from generate_cultivation_guides import main as generate_guides
        from generate_disease_knowledge import main as generate_diseases

        print("Step 1: Generating cultivation guides for all 96 crops...")
        generate_guides()
        print()

        print("Step 2: Generating disease knowledge base...")
        generate_diseases()
        print()

        print("=" * 80)
        print("✅ EDGE AI SETUP COMPLETE!")
        print("=" * 80)
        print()
        print("Knowledge bases generated:")
        print("  - Cultivation guides for 96 crops")
        print("  - Disease knowledge for all crops")
        print()
        print("You can now use the edge AI chatbot and vision models!")

    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("Make sure all required files are in the same directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
