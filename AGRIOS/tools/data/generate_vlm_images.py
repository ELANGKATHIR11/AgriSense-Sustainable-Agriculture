"""
Generate simple placeholder JPG images for VLM pairs listed in
`agrisense_app/backend/data/vlm_pairs.csv`.
Creates images under `agrisense_app/backend/data/images/`.
"""
from pathlib import Path
import csv
from PIL import Image, ImageDraw, ImageFont

CSV = Path("agrisense_app/backend/data/vlm_pairs.csv")
OUT_DIR = Path("agrisense_app/backend/data/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1024, 768
BG_COLORS = [(200, 230, 201), (255, 224, 178), (187, 222, 251), (255, 204, 188), (232, 234, 246)]

def make_placeholder(text, outpath, color):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    # Draw centered text (crop name)
    lines = []
    # shorten if too long
    if len(text) > 40:
        text = text[:37] + "..."
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((WIDTH - w) / 2, (HEIGHT - h) / 2), text, fill=(30,30,30), font=font)
    img.save(outpath, quality=85)


def main():
    if not CSV.exists():
        print(f"vLM pairs CSV not found: {CSV}")
        return
    rows = []
    with CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    count = 0
    for i, r in enumerate(rows):
        image_path = Path(r.get("image_path", "images/unknown.jpg"))
        name = image_path.name
        outpath = OUT_DIR / name
        caption = r.get("caption", "AgriSense placeholder")
        color = BG_COLORS[i % len(BG_COLORS)]
        make_placeholder(caption, outpath, color)
        count += 1
    print(f"✅ Generated {count} placeholder images to {OUT_DIR}")

if __name__ == '__main__':
    main()
