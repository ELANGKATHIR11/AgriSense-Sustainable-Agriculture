"""
Synthesize Crop_recommendation.csv from india_crop_dataset.csv
Produces columns: N,P,K,temperature,humidity,pH,rainfall,label
"""
import csv
import random
from pathlib import Path
import math

INPUT = Path("agrisense_app/backend/india_crop_dataset.csv")
OUTPUT_DIR = Path("agrisense_app/backend/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTPUT_DIR / "Crop_recommendation.csv"

SAMPLES_PER_CROP = 100
random.seed(42)

def float_round(x, decimals=2):
    return round(float(x), decimals)


def read_crops():
    crops = []
    with INPUT.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            crops.append(r)
    return crops


def sample_for_crop(crop_row, n=SAMPLES_PER_CROP):
    samples = []
    # Parse numeric ranges with fallbacks
    def parse(key):
        v = crop_row.get(key, "")
        try:
            return float(v)
        except Exception:
            return None

    pH_min = parse("pH_Min") or 5.5
    pH_max = parse("pH_Max") or 7.5
    ph_opt = parse("pH_Optimal") or (pH_min + pH_max) / 2

    n_min = parse("Nitrogen_Min_kg_ha") or 20
    n_max = parse("Nitrogen_Max_kg_ha") or 200
    n_opt = parse("Nitrogen_Optimal_kg_ha") or (n_min + n_max) / 2

    p_min = parse("Phosphorus_Min_kg_ha") or 10
    p_max = parse("Phosphorus_Max_kg_ha") or 100
    p_opt = parse("Phosphorus_Optimal_kg_ha") or (p_min + p_max) / 2

    k_min = parse("Potassium_Min_kg_ha") or 10
    k_max = parse("Potassium_Max_kg_ha") or 300
    k_opt = parse("Potassium_Optimal_kg_ha") or (k_min + k_max) / 2

    temp_min = parse("Temperature_Min_C") or 10
    temp_max = parse("Temperature_Max_C") or 35
    temp_opt = parse("Temperature_Optimal_C") or (temp_min + temp_max) / 2

    humidity_min = parse("Humidity_Min_percent") or 40
    humidity_max = parse("Humidity_Max_percent") or 90
    humidity_opt = parse("Humidity_Optimal_percent") or (humidity_min + humidity_max) / 2

    rainfall_req = parse("Rainfall_Requirement_mm") or 400

    for i in range(n):
        # Sample values biased towards optimal but with variance
        def biased_sample(opt, low, high, sigma_ratio=0.12):
            sigma = max(0.01, (high - low) * sigma_ratio)
            val = random.gauss(opt, sigma)
            # clamp
            val = max(low, min(high, val))
            return float_round(val, 2)

        ph = biased_sample(ph_opt, pH_min, pH_max, sigma_ratio=0.08)
        nitrogen = biased_sample(n_opt, n_min, n_max, sigma_ratio=0.2)
        phosphorus = biased_sample(p_opt, p_min, p_max, sigma_ratio=0.2)
        potassium = biased_sample(k_opt, k_min, k_max, sigma_ratio=0.2)
        temperature = biased_sample(temp_opt, temp_min, temp_max, sigma_ratio=0.12)
        humidity = biased_sample(humidity_opt, humidity_min, humidity_max, sigma_ratio=0.12)
        # rainfall as fraction of requirement with seasonal noise
        rainfall = float_round(random.normalvariate(rainfall_req * 0.9, rainfall_req * 0.15), 1)
        if rainfall < 0:
            rainfall = float_round(abs(rainfall_req * random.random()), 1)

        samples.append({
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "label": crop_row.get("Crop", "Unknown")
        })
    return samples


def synthesize():
    crops = read_crops()
    all_samples = []
    for crop in crops:
        samples = sample_for_crop(crop, n=SAMPLES_PER_CROP)
        all_samples.extend(samples)

    # Write CSV
    fieldnames = ["N","P","K","temperature","humidity","ph","rainfall","label"]
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_samples:
            writer.writerow(row)
    print(f"✅ Generated {len(all_samples)} samples to {OUTPUT}")


if __name__ == "__main__":
    synthesize()
