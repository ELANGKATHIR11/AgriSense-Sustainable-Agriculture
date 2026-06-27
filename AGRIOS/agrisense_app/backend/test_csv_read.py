import os
import csv

def test_read():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(ROOT, "india_crop_dataset.csv")
    print(f"Looking for dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("File NOT found!")
        return

    print("File found.")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"Read {len(rows)} rows.")
            if rows:
                print("First row keys:", rows[0].keys())
                print("First row values:", rows[0])
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    test_read()
