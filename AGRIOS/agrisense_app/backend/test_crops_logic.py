
import os
import sys

# Ensure we can import from the directory
sys.path.append(os.path.dirname(__file__))

try:
    from main import _dataset_to_cards, get_crops_full
    
    print("Testing _dataset_to_cards...")
    cards = _dataset_to_cards()
    print(f"Count: {len(cards)}")
    if len(cards) > 0:
        print(f"First card: {cards[0].name}")
    else:
        print("No cards returned.")
        
    # Check paths manually
    ROOT = os.path.dirname(os.path.abspath("main.py")) # Approximating main.py location
    print(f"Calculated ROOT for verification: {ROOT}")
    
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
