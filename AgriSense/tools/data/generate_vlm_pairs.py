"""
Generate VLM pairs CSV placeholders for AgriSense
Produces CSV: agrisense_app/backend/data/vlm_pairs.csv
Columns: image_path,caption
Image files are placeholders (not created) named images/<crop>.jpg
"""
import csv
from pathlib import Path

INPUT = Path("agrisense_app/backend/india_crop_dataset.csv")
OUTPUT_DIR = Path("agrisense_app/backend/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTPUT_DIR / "vlm_pairs.csv"


def read_crops():
    crops = []
    with INPUT.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            crops.append(r)
    return crops


def generate():
    crops = read_crops()
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "caption"])
        writer.writeheader()
        for crop in crops:
            name = crop.get("Crop", "unknown").replace(" ", "_")
            soil = crop.get("Soil_Type", "mixed")
            caption = f"A healthy {crop.get('Crop')} growing in a {soil} field; close-up of leaves and soil conditions."
            image_path = f"images/{name}.jpg"
            writer.writerow({"image_path": image_path, "caption": caption})
    print(f"✅ Generated {len(crops)} VLM pairs to {OUTPUT}")

if __name__ == "__main__":
    generate()
