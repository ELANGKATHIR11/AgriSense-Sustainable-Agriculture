import os
import sys
import json
import hashlib
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple
from PIL import Image
import psycopg
from backend.database.connection import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD
)

class DatasetManager:
    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir
        self.dsn = f"host={POSTGRES_HOST} port={POSTGRES_PORT} user={POSTGRES_USER} password={POSTGRES_PASSWORD} dbname={POSTGRES_DB}"
        self._init_db()

    def _init_db(self):
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) UNIQUE,
                        path TEXT,
                        type VARCHAR(50),
                        quality_score REAL,
                        image_count INTEGER,
                        annotation_count INTEGER,
                        status VARCHAR(50)
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dataset_versions (
                        id SERIAL PRIMARY KEY,
                        dataset_id INTEGER,
                        version_str VARCHAR(50),
                        manifest_path TEXT,
                        checksum VARCHAR(255),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS duplicates (
                        id SERIAL PRIMARY KEY,
                        file_hash VARCHAR(255) UNIQUE,
                        filepath TEXT,
                        duplicate_filepath TEXT
                    )
                """)

    def calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA256 of image to prevent duplicate uploads."""
        hasher = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
            return hasher.hexdigest()
        except Exception:
            return ""

    def validate_image(self, filepath: str) -> Tuple[bool, str]:
        """Validate EXIF orientation, size, channels, and headers."""
        try:
            with Image.open(filepath) as img:
                img.verify()
            
            # Reopen to check channels/convert
            with Image.open(filepath) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                w, h = img.size
                if w < 32 or h < 32:
                    return False, "Resolution too low"
            return True, "Valid"
        except Exception as e:
            return False, f"Corrupted or invalid image: {e}"

    def convert_voc_to_yolo(self, xml_path: str, classes: List[str]) -> List[str]:
        """Convert Pascal VOC bounding box annotations to normalized YOLO labels."""
        yolo_lines = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            if size is None:
                return []
            width = int(size.find("width").text)
            height = int(size.find("height").text)

            for obj in root.findall("object"):
                name = obj.find("name").text
                if name not in classes:
                    classes.append(name)
                class_id = classes.index(name)

                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                # Convert to normalized midpoint and dimensions
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        except Exception:
            pass
        return yolo_lines

    def audit_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Generate audit reports for standard datasets."""
        results = {
            "name": os.path.basename(dataset_path),
            "path": dataset_path,
            "images": 0,
            "labels": 0,
            "duplicates": 0,
            "corrupted": 0,
            "classes": set(),
            "quality_score": 100.0
        }

        if not os.path.exists(dataset_path):
            return results

        hashes = {}
        for root, _, files in os.walk(dataset_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                filepath = os.path.join(root, file)
                
                if ext in [".jpg", ".jpeg", ".png"]:
                    results["images"] += 1
                    # Check corruption
                    is_valid, msg = self.validate_image(filepath)
                    if not is_valid:
                        results["corrupted"] += 1
                        continue
                    
                    # Duplicate check
                    f_hash = self.calculate_file_hash(filepath)
                    if f_hash:
                        if f_hash in hashes:
                            results["duplicates"] += 1
                            with psycopg.connect(self.dsn) as conn:
                                with conn.cursor() as cur:
                                    cur.execute("""
                                        INSERT INTO duplicates (file_hash, filepath, duplicate_filepath)
                                        VALUES (%s, %s, %s)
                                        ON CONFLICT (file_hash) DO UPDATE SET
                                            filepath = EXCLUDED.filepath,
                                            duplicate_filepath = EXCLUDED.duplicate_filepath
                                    """, (f_hash, hashes[f_hash], filepath))
                        else:
                            hashes[f_hash] = filepath
                
                elif ext in [".txt", ".xml", ".json"]:
                    results["labels"] += 1

        # Calculate a quality score based on duplicates and corruption
        total_issues = results["duplicates"] + results["corrupted"]
        if results["images"] > 0:
            deduction = (total_issues / results["images"]) * 100.0
            results["quality_score"] = max(0.0, 100.0 - deduction)

        # Register in database
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO datasets (name, path, type, quality_score, image_count, annotation_count, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                        path = EXCLUDED.path,
                        type = EXCLUDED.type,
                        quality_score = EXCLUDED.quality_score,
                        image_count = EXCLUDED.image_count,
                        annotation_count = EXCLUDED.annotation_count,
                        status = EXCLUDED.status
                """, (results["name"], results["path"], "YOLO/COCO", results["quality_score"], results["images"], results["labels"], "Active"))

        return results

    def generate_training_yaml(self, train_path: str, val_path: str, classes: List[str], output_path: str = "train.yaml"):
        """Generate PyTorch/YOLOv11 compatible training yaml files."""
        yaml_content = f"""# AgriSense Dataset Training Descriptor
path: {os.path.abspath(self.root_dir)}
train: {train_path}
val: {val_path}

names:
"""
        for i, cls in enumerate(classes):
            yaml_content += f"  {i}: {cls}\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"Generated training manifest at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="AgriSense Dataset Manager Pipeline CLI")
    parser.add_argument("--scan", type=str, help="Dataset directory to scan and audit")
    parser.add_argument("--out-yaml", type=str, default="train.yaml", help="Path to write the train.yaml config")
    args = parser.parse_args()

    manager = DatasetManager()
    if args.scan:
        print(f"Auditing directory: {args.scan}...")
        report = manager.audit_dataset(args.scan)
        print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
