from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("f:/AGRISENSEFULL-STACK/backend/ml/datasets_enhanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Full list of 96 crops
ALL_CROPS = [
    "Almond",
    "Apple",
    "Arecanut",
    "Arhar",
    "Bajra",
    "Banana",
    "Barley",
    "Barnyard_Millet",
    "Beetroot",
    "Bitter_Gourd",
    "Black_Pepper",
    "Bottle_Gourd",
    "Brinjal",
    "Buckwheat",
    "Cabbage",
    "Cardamom",
    "Carrot",
    "Cashew",
    "Castor",
    "Cauliflower",
    "Chickpea",
    "Chilli",
    "Cluster_Bean",
    "Coconut",
    "Coffee",
    "Coriander",
    "Cotton",
    "Cucumber",
    "Cumin",
    "Custard_Apple",
    "Dragon_Fruit",
    "Fenugreek",
    "Field_Pea",
    "Foxtail_Millet",
    "French_Bean",
    "Garlic",
    "Ginger",
    "Grapes",
    "Green_Pea",
    "Groundnut",
    "Guava",
    "Horse_Gram",
    "Jackfruit",
    "Jowar",
    "Jute",
    "Kidney_Bean",
    "Kodo_Millet",
    "Lentil",
    "Lettuce",
    "Linseed",
    "Litchi",
    "Little_Millet",
    "Maize",
    "Mango",
    "Masoor",
    "Moong",
    "Moth_Bean",
    "Muskmelon",
    "Mustard",
    "Niger",
    "Oats",
    "Okra",
    "Onion",
    "Orange",
    "Papaya",
    "Pearl_Millet",
    "Pigeon_Pea",
    "Pineapple",
    "Pomegranate",
    "Potato",
    "Proso_Millet",
    "Pumpkin",
    "Radish",
    "Ragi",
    "Rice",
    "Ridge_Gourd",
    "Rubber",
    "Safflower",
    "Sapota",
    "Sesame",
    "Sorghum",
    "Soybean",
    "Spinach",
    "Strawberry",
    "Sugarcane",
    "Sunflower",
    "Sweet_Potato",
    "Tea",
    "Tobacco",
    "Tomato",
    "Turmeric",
    "Turnip",
    "Urad",
    "Walnut",
    "Watermelon",
    "Wheat",
]

CROP_MAP = {
    "Rice": "cereal",
    "Wheat": "cereal",
    "Maize": "cereal",
    "Barley": "cereal",
    "Oats": "cereal",
    "Buckwheat": "cereal",
    "Bajra": "millet",
    "Jowar": "millet",
    "Ragi": "millet",
    "Sorghum": "millet",
    "Barnyard_Millet": "millet",
    "Foxtail_Millet": "millet",
    "Kodo_Millet": "millet",
    "Little_Millet": "millet",
    "Pearl_Millet": "millet",
    "Proso_Millet": "millet",
    "Chickpea": "pulse",
    "Pigeon_Pea": "pulse",
    "Arhar": "pulse",
    "Moong": "pulse",
    "Masoor": "pulse",
    "Urad": "pulse",
    "Lentil": "pulse",
    "Kidney_Bean": "pulse",
    "Moth_Bean": "pulse",
    "Horse_Gram": "pulse",
    "Cluster_Bean": "pulse",
    "French_Bean": "pulse",
    "Green_Pea": "pulse",
    "Field_Pea": "pulse",
    "Soybean": "pulse",
    "Tomato": "vegetable",
    "Brinjal": "vegetable",
    "Chilli": "vegetable",
    "Okra": "vegetable",
    "Cabbage": "vegetable",
    "Cauliflower": "vegetable",
    "Spinach": "vegetable",
    "Lettuce": "vegetable",
    "Pumpkin": "vegetable",
    "Bottle_Gourd": "vegetable",
    "Bitter_Gourd": "vegetable",
    "Ridge_Gourd": "vegetable",
    "Cucumber": "vegetable",
    "Drumstick": "vegetable",
    "Potato": "tuber",
    "Onion": "tuber",
    "Garlic": "tuber",
    "Ginger": "tuber",
    "Turmeric": "tuber",
    "Radish": "tuber",
    "Carrot": "tuber",
    "Beetroot": "tuber",
    "Sweet_Potato": "tuber",
    "Turnip": "tuber",
    "Mango": "fruit",
    "Banana": "fruit",
    "Papaya": "fruit",
    "Guava": "fruit",
    "Pomegranate": "fruit",
    "Sapota": "fruit",
    "Jackfruit": "fruit",
    "Custard_Apple": "fruit",
    "Litchi": "fruit",
    "Pineapple": "fruit",
    "Dragon_Fruit": "fruit",
    "Watermelon": "fruit",
    "Muskmelon": "fruit",
    "Apple": "cold",
    "Orange": "cold",
    "Grapes": "cold",
    "Strawberry": "cold",
    "Walnut": "cold",
    "Almond": "cold",
    "Sugarcane": "plantation",
    "Cotton": "fiber",
    "Jute": "fiber",
    "Rubber": "plantation",
    "Tea": "plantation",
    "Coffee": "plantation",
    "Coconut": "plantation",
    "Arecanut": "plantation",
    "Cashew": "plantation",
    "Tobacco": "plantation",
    "Black_Pepper": "spice",
    "Cardamom": "spice",
    "Clove": "spice",
    "Cumin": "spice",
    "Coriander": "spice",
    "Fenugreek": "spice",
    "Groundnut": "oilseed",
    "Mustard": "oilseed",
    "Sunflower": "oilseed",
    "Sesame": "oilseed",
    "Castor": "oilseed",
    "Linseed": "oilseed",
    "Safflower": "oilseed",
    "Niger": "oilseed",
}

# DNA for Species Precision
CROP_DNA = {
    crop: {
        "N_off": i * 5,
        "P_off": i * 3,
        "K_off": i * 4,
        "T_off": i * 0.2,
        "yield_base": 1.5 + (i % 5) * 0.5,
    }
    for i, crop in enumerate(ALL_CROPS)
}

# 1. CROP DATA
crop_samples = []
for crop in ALL_CROPS:
    dna = CROP_DNA[crop]
    for _ in range(120):
        crop_samples.append(
            {
                "label": crop,
                "N": np.random.normal(20 + dna["N_off"], 1),
                "P": np.random.normal(20 + dna["P_off"], 1),
                "K": np.random.normal(20 + dna["K_off"], 1),
                "temperature": np.random.normal(22 + dna["T_off"], 0.5),
                "humidity": np.random.normal(60, 2),
                "ph": np.random.normal(6.5, 0.1),
                "rainfall": np.random.normal(100, 5),
            }
        )
pd.DataFrame(crop_samples).to_csv(
    OUTPUT_DIR / "crop_recommendation/crop_data_enhanced.csv", index=False
)

# 2. WATER DATA
water_samples = []
for crop in ALL_CROPS:
    for _ in range(50):
        t, h, r = (
            np.random.uniform(15, 40),
            np.random.uniform(20, 90),
            np.random.uniform(0, 15),
        )
        eto = (0.0023 * t + 0.3) * (0.6 + 0.4 * (1 - h / 100)) * 5
        water_samples.append(
            {
                "crop": crop,
                "temperature": round(t, 2),
                "humidity": round(h, 2),
                "rainfall": round(r, 2),
                "area": 1.0,
                "water_requirement": round(eto * 0.9, 2),
            }
        )
pd.DataFrame(water_samples).to_csv(
    OUTPUT_DIR / "water_requirement/water_requirement_enhanced.csv", index=False
)

# 3. YIELD DATA (Stage 4 Target)
yield_samples = []
for crop in ALL_CROPS:
    dna = CROP_DNA[crop]
    for _ in range(150):
        area = np.random.uniform(0.5, 10.0)
        n, p, k = (
            np.random.normal(20 + dna["N_off"], 2),
            np.random.normal(20 + dna["P_off"], 2),
            np.random.normal(20 + dna["K_off"], 2),
        )
        t, h, r = (
            np.random.normal(22 + dna["T_off"], 1),
            np.random.normal(60, 5),
            np.random.normal(100, 10),
        )
        # Physics: Yield depends on Area * Base * (NPK utilization)
        multiplier = (
            (n / (20 + dna["N_off"]))
            * (p / (20 + dna["P_off"]))
            * (k / (20 + dna["K_off"]))
            * (1 - abs(t - (22 + dna["T_off"])) / 30)
        )
        prod = area * dna["yield_base"] * multiplier
        yield_samples.append(
            {
                "crop": crop,
                "N": n,
                "P": p,
                "K": k,
                "temperature": t,
                "humidity": h,
                "rainfall": r,
                "area": area,
                "production": round(max(0, prod), 2),
            }
        )
y_path = OUTPUT_DIR / "yield_prediction"
y_path.mkdir(exist_ok=True)
pd.DataFrame(yield_samples).to_csv(y_path / "yield_data_enhanced.csv", index=False)

# 4. SEASON
season_samples = []
for s in ["Summer", "Winter", "Monsoon", "Spring", "Autumn"]:
    t, h, r = (
        (35, 40, 20)
        if s == "Summer"
        else (
            (15, 60, 30)
            if s == "Winter"
            else (
                (28, 85, 250)
                if s == "Monsoon"
                else (25, 55, 50) if s == "Spring" else (24, 75, 80)
            )
        )
    )
    for _ in range(400):
        season_samples.append(
            {
                "temperature": np.random.normal(t, 1),
                "humidity": np.random.normal(h, 1),
                "rainfall": np.random.normal(r, 2),
                "label": s,
            }
        )
pd.DataFrame(season_samples).to_csv(
    OUTPUT_DIR / "season_classification/season_data_enhanced.csv", index=False
)

print("🚀 4-STAGE Industrial Datasets Locked!")
