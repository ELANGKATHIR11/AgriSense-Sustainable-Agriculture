import os
import csv
import random
import pandas as pd
from pathlib import Path

# =============================================================================
# DATASET GENERATOR: Synthetic Soil Health Data
# This serves as a high-quality fallback for the AgriSense project
# =============================================================================

CSV_FILE = "soil_health_dataset.csv"
NUM_SAMPLES = 1000  # Generate 1000 realistic records

# Typical ranges for Indian soil (standard SHC parameters)
SOIL_PROFILES = {
    "Alluvial": {
        "pH": (6.5, 8.5),
        "EC": (0.1, 0.8),
        "OC": (0.4, 0.9),
        "N": (200, 450),
        "P": (10, 25),
        "K": (150, 350),
        "crops": ["Rice", "Wheat", "Sugarcane", "Cotton"],
    },
    "Black": {
        "pH": (7.5, 8.5),
        "EC": (0.2, 0.6),
        "OC": (0.3, 0.7),
        "N": (150, 300),
        "P": (8, 20),
        "K": (200, 500),
        "crops": ["Cotton", "Soybean", "Pulses", "Wheat"],
    },
    "Red": {
        "pH": (5.5, 7.0),
        "EC": (0.05, 0.3),
        "OC": (0.2, 0.6),
        "N": (100, 250),
        "P": (5, 15),
        "K": (100, 300),
        "crops": ["Millets", "Groundnut", "Pulses", "Tobacco"],
    },
    "Laterite": {
        "pH": (4.5, 6.0),
        "EC": (0.01, 0.2),
        "OC": (0.5, 1.2),
        "N": (250, 500),
        "P": (2, 10),
        "K": (50, 200),
        "crops": ["Tea", "Coffee", "Cashew", "Rubber"],
    },
}

# Crop groups for environmental factors
CROP_MAP = {
    "Rice": "cereal",
    "Wheat": "cereal",
    "Sugarcane": "plantation",
    "Cotton": "fiber",
    "Soybean": "pulse",
    "Pulses": "pulse",
    "Millets": "millet",
    "Groundnut": "oilseed",
    "Tobacco": "plantation",
    "Tea": "plantation",
    "Coffee": "plantation",
    "Cashew": "plantation",
    "Rubber": "plantation",
}

# Environmental factors by crop type (approximate)
ENV_PROFILES = {
    "cereal": {"temp": (20, 30), "hum": (60, 90), "rain": (150, 300)},
    "pulse": {"temp": (15, 25), "hum": (40, 70), "rain": (50, 150)},
    "millet": {"temp": (25, 35), "hum": (30, 60), "rain": (30, 100)},
    "plantation": {"temp": (20, 35), "hum": (70, 95), "rain": (200, 500)},
    "fiber": {"temp": (25, 40), "hum": (50, 80), "rain": (100, 250)},
    "oilseed": {"temp": (20, 30), "hum": (40, 70), "rain": (40, 120)},
}

MICRONUTRIENTS = {
    "Fe": (2.0, 10.0),
    "Zn": (0.3, 2.0),
    "Mn": (2.0, 15.0),
    "Cu": (0.2, 1.5),
    "B": (0.1, 1.0),
}


def generate_realistic_sample():
    profile_name = random.choice(list(SOIL_PROFILES.keys()))
    profile = SOIL_PROFILES[profile_name]

    crop = random.choice(profile["crops"])
    crop_type = CROP_MAP.get(crop, "pulse")
    env = ENV_PROFILES[crop_type]

    data = {
        "N": round(random.uniform(*profile["N"]), 2),
        "P": round(random.uniform(*profile["P"]), 2),
        "K": round(random.uniform(*profile["K"]), 2),
        "temperature": round(random.uniform(*env["temp"]), 2),
        "humidity": round(random.uniform(*env["hum"]), 2),
        "ph": round(random.uniform(*profile["pH"]), 2),
        "rainfall": round(random.uniform(*env["rain"]), 2),
        "label": crop.lower(),
    }

    # Add extra soil health parameters
    data["EC"] = round(random.uniform(*profile["EC"]), 2)
    data["OC"] = round(random.uniform(*profile["OC"]), 2)
    for m, r in MICRONUTRIENTS.items():
        data[m] = round(random.uniform(*r), 2)

    return data


def main():
    print("🌾 AgriSense Soil & Environment Dataset Generator")
    print("=" * 50)
    print(f"Generating {NUM_SAMPLES} records for model retraining...")

    dataset = [generate_realistic_sample() for _ in range(NUM_SAMPLES)]
    df = pd.DataFrame(dataset)

    # Standard column order for Crop Recommendation model compatibility
    std_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    extra_cols = ["EC", "OC", "Fe", "Zn", "Mn", "Cu", "B"]
    df = df[std_cols + extra_cols]

    df.to_csv(CSV_FILE, index=False)

    print(f"\n✅ Dataset created: {CSV_FILE}")
    print("\n📊 Data Summary:")
    print(df.describe().T[["mean", "min", "max"]])
    print("\n📦 Target distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
