"""
Generate Comprehensive 100-Crop Dataset for AgriSense ML
Includes diverse Indian crops with scientifically accurate agricultural parameters
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define 100 crops with accurate agricultural data
crops_data = [
    # Cereals (15 crops)
    {"crop": "rice", "N": 120, "P": 60, "K": 40, "temp": 25, "humidity": 80, "ph": 6.5, "rainfall": 1200, "type": "cereal", "season": "kharif", "days": 120},
    {"crop": "wheat", "N": 100, "P": 50, "K": 50, "temp": 22, "humidity": 55, "ph": 6.8, "rainfall": 500, "type": "cereal", "season": "rabi", "days": 140},
    {"crop": "maize", "N": 110, "P": 55, "K": 50, "temp": 24, "humidity": 65, "ph": 6.5, "rainfall": 600, "type": "cereal", "season": "kharif", "days": 90},
    {"crop": "bajra", "N": 60, "P": 40, "K": 30, "temp": 28, "humidity": 50, "ph": 6.5, "rainfall": 400, "type": "cereal", "season": "kharif", "days": 75},
    {"crop": "jowar", "N": 80, "P": 40, "K": 40, "temp": 26, "humidity": 55, "ph": 6.8, "rainfall": 500, "type": "cereal", "season": "kharif", "days": 100},
    {"crop": "ragi", "N": 50, "P": 40, "K": 30, "temp": 24, "humidity": 60, "ph": 6.5, "rainfall": 400, "type": "cereal", "season": "kharif", "days": 120},
    {"crop": "barley", "N": 60, "P": 30, "K": 30, "temp": 18, "humidity": 50, "ph": 7.0, "rainfall": 400, "type": "cereal", "season": "rabi", "days": 120},
    {"crop": "oats", "N": 80, "P": 40, "K": 40, "temp": 20, "humidity": 55, "ph": 6.5, "rainfall": 450, "type": "cereal", "season": "rabi", "days": 110},
    {"crop": "pearl_millet", "N": 60, "P": 35, "K": 30, "temp": 27, "humidity": 52, "ph": 6.8, "rainfall": 400, "type": "cereal", "season": "kharif", "days": 80},
    {"crop": "foxtail_millet", "N": 50, "P": 30, "K": 25, "temp": 25, "humidity": 55, "ph": 6.5, "rainfall": 350, "type": "cereal", "season": "kharif", "days": 70},
    {"crop": "kodo_millet", "N": 40, "P": 25, "K": 20, "temp": 24, "humidity": 58, "ph": 6.2, "rainfall": 400, "type": "cereal", "season": "kharif", "days": 90},
    {"crop": "little_millet", "N": 45, "P": 28, "K": 22, "temp": 26, "humidity": 56, "ph": 6.5, "rainfall": 380, "type": "cereal", "season": "kharif", "days": 85},
    {"crop": "proso_millet", "N": 55, "P": 32, "K": 28, "temp": 25, "humidity": 54, "ph": 6.8, "rainfall": 350, "type": "cereal", "season": "kharif", "days": 75},
    {"crop": "barnyard_millet", "N": 48, "P": 30, "K": 24, "temp": 23, "humidity": 57, "ph": 6.5, "rainfall": 370, "type": "cereal", "season": "kharif", "days": 80},
    {"crop": "sorghum", "N": 85, "P": 42, "K": 42, "temp": 27, "humidity": 56, "ph": 6.7, "rainfall": 520, "type": "cereal", "season": "kharif", "days": 105},
    
    # Pulses (15 crops)
    {"crop": "chickpea", "N": 30, "P": 60, "K": 40, "temp": 22, "humidity": 50, "ph": 7.0, "rainfall": 400, "type": "pulse", "season": "rabi", "days": 130},
    {"crop": "pigeon_pea", "N": 25, "P": 50, "K": 40, "temp": 26, "humidity": 60, "ph": 6.5, "rainfall": 600, "type": "pulse", "season": "kharif", "days": 180},
    {"crop": "moong", "N": 25, "P": 50, "K": 35, "temp": 28, "humidity": 65, "ph": 6.8, "rainfall": 500, "type": "pulse", "season": "kharif", "days": 70},
    {"crop": "urad", "N": 25, "P": 55, "K": 40, "temp": 27, "humidity": 62, "ph": 6.5, "rainfall": 550, "type": "pulse", "season": "kharif", "days": 75},
    {"crop": "masoor", "N": 20, "P": 50, "K": 35, "temp": 21, "humidity": 55, "ph": 7.0, "rainfall": 450, "type": "pulse", "season": "rabi", "days": 140},
    {"crop": "arhar", "N": 25, "P": 52, "K": 42, "temp": 26, "humidity": 61, "ph": 6.6, "rainfall": 580, "type": "pulse", "season": "kharif", "days": 175},
    {"crop": "kidney_bean", "N": 30, "P": 55, "K": 45, "temp": 23, "humidity": 58, "ph": 6.8, "rainfall": 500, "type": "pulse", "season": "kharif", "days": 90},
    {"crop": "horse_gram", "N": 20, "P": 45, "K": 30, "temp": 25, "humidity": 54, "ph": 6.5, "rainfall": 400, "type": "pulse", "season": "kharif", "days": 120},
    {"crop": "moth_bean", "N": 22, "P": 48, "K": 32, "temp": 28, "humidity": 52, "ph": 7.0, "rainfall": 350, "type": "pulse", "season": "kharif", "days": 75},
    {"crop": "field_pea", "N": 28, "P": 52, "K": 38, "temp": 20, "humidity": 56, "ph": 6.8, "rainfall": 450, "type": "pulse", "season": "rabi", "days": 120},
    {"crop": "lentil", "N": 22, "P": 50, "K": 35, "temp": 21, "humidity": 54, "ph": 7.0, "rainfall": 420, "type": "pulse", "season": "rabi", "days": 135},
    {"crop": "green_pea", "N": 32, "P": 58, "K": 42, "temp": 19, "humidity": 60, "ph": 6.5, "rainfall": 500, "type": "pulse", "season": "rabi", "days": 90},
    {"crop": "french_bean", "N": 35, "P": 60, "K": 48, "temp": 22, "humidity": 62, "ph": 6.8, "rainfall": 550, "type": "pulse", "season": "kharif", "days": 60},
    {"crop": "cluster_bean", "N": 25, "P": 48, "K": 35, "temp": 26, "humidity": 55, "ph": 7.2, "rainfall": 400, "type": "pulse", "season": "kharif", "days": 80},
    {"crop": "cowpea", "N": 28, "P": 52, "K": 38, "temp": 27, "humidity": 60, "ph": 6.5, "rainfall": 480, "type": "pulse", "season": "kharif", "days": 70},
    
    # Vegetables (20 crops)
    {"crop": "potato", "N": 150, "P": 80, "K": 100, "temp": 22, "humidity": 70, "ph": 6.0, "rainfall": 500, "type": "vegetable", "season": "rabi", "days": 90},
    {"crop": "tomato", "N": 120, "P": 70, "K": 80, "temp": 24, "humidity": 65, "ph": 6.5, "rainfall": 600, "type": "vegetable", "season": "kharif", "days": 75},
    {"crop": "onion", "N": 100, "P": 60, "K": 70, "temp": 23, "humidity": 60, "ph": 6.5, "rainfall": 450, "type": "vegetable", "season": "rabi", "days": 120},
    {"crop": "cabbage", "N": 110, "P": 65, "K": 75, "temp": 20, "humidity": 68, "ph": 6.5, "rainfall": 500, "type": "vegetable", "season": "rabi", "days": 80},
    {"crop": "cauliflower", "N": 115, "P": 68, "K": 78, "temp": 19, "humidity": 70, "ph": 6.5, "rainfall": 550, "type": "vegetable", "season": "rabi", "days": 90},
    {"crop": "brinjal", "N": 100, "P": 60, "K": 65, "temp": 25, "humidity": 68, "ph": 6.5, "rainfall": 600, "type": "vegetable", "season": "kharif", "days": 100},
    {"crop": "chilli", "N": 90, "P": 55, "K": 60, "temp": 26, "humidity": 65, "ph": 6.5, "rainfall": 650, "type": "vegetable", "season": "kharif", "days": 110},
    {"crop": "okra", "N": 70, "P": 50, "K": 55, "temp": 27, "humidity": 62, "ph": 6.5, "rainfall": 500, "type": "vegetable", "season": "kharif", "days": 60},
    {"crop": "carrot", "N": 80, "P": 55, "K": 60, "temp": 21, "humidity": 65, "ph": 6.5, "rainfall": 450, "type": "vegetable", "season": "rabi", "days": 90},
    {"crop": "radish", "N": 75, "P": 50, "K": 55, "temp": 22, "humidity": 65, "ph": 6.5, "rainfall": 400, "type": "vegetable", "season": "rabi", "days": 40},
    {"crop": "pumpkin", "N": 100, "P": 60, "K": 70, "temp": 24, "humidity": 70, "ph": 6.5, "rainfall": 600, "type": "vegetable", "season": "kharif", "days": 100},
    {"crop": "bottle_gourd", "N": 85, "P": 55, "K": 60, "temp": 25, "humidity": 68, "ph": 6.5, "rainfall": 550, "type": "vegetable", "season": "kharif", "days": 80},
    {"crop": "bitter_gourd", "N": 80, "P": 50, "K": 55, "temp": 26, "humidity": 66, "ph": 6.5, "rainfall": 500, "type": "vegetable", "season": "kharif", "days": 70},
    {"crop": "ridge_gourd", "N": 75, "P": 48, "K": 52, "temp": 25, "humidity": 67, "ph": 6.5, "rainfall": 520, "type": "vegetable", "season": "kharif", "days": 75},
    {"crop": "cucumber", "N": 70, "P": 45, "K": 50, "temp": 24, "humidity": 68, "ph": 6.5, "rainfall": 500, "type": "vegetable", "season": "kharif", "days": 60},
    {"crop": "spinach", "N": 90, "P": 55, "K": 60, "temp": 20, "humidity": 65, "ph": 6.5, "rainfall": 450, "type": "vegetable", "season": "rabi", "days": 45},
    {"crop": "beetroot", "N": 85, "P": 52, "K": 58, "temp": 21, "humidity": 66, "ph": 6.5, "rainfall": 480, "type": "vegetable", "season": "rabi", "days": 75},
    {"crop": "turnip", "N": 80, "P": 50, "K": 55, "temp": 20, "humidity": 64, "ph": 6.5, "rainfall": 460, "type": "vegetable", "season": "rabi", "days": 70},
    {"crop": "lettuce", "N": 75, "P": 48, "K": 52, "temp": 18, "humidity": 70, "ph": 6.5, "rainfall": 500, "type": "vegetable", "season": "rabi", "days": 50},
    {"crop": "sweet_potato", "N": 95, "P": 58, "K": 65, "temp": 25, "humidity": 68, "ph": 6.0, "rainfall": 550, "type": "vegetable", "season": "kharif", "days": 120},
    
    # Fruits (15 crops)
    {"crop": "mango", "N": 100, "P": 60, "K": 80, "temp": 27, "humidity": 65, "ph": 6.5, "rainfall": 1000, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "banana", "N": 150, "P": 80, "K": 120, "temp": 28, "humidity": 75, "ph": 6.5, "rainfall": 1500, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "papaya", "N": 120, "P": 70, "K": 90, "temp": 26, "humidity": 70, "ph": 6.5, "rainfall": 1200, "type": "fruit", "season": "perennial", "days": 300},
    {"crop": "guava", "N": 90, "P": 55, "K": 70, "temp": 26, "humidity": 68, "ph": 6.5, "rainfall": 900, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "apple", "N": 80, "P": 50, "K": 60, "temp": 18, "humidity": 60, "ph": 6.5, "rainfall": 800, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "grapes", "N": 100, "P": 60, "K": 80, "temp": 24, "humidity": 55, "ph": 6.5, "rainfall": 600, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "orange", "N": 110, "P": 65, "K": 85, "temp": 25, "humidity": 65, "ph": 6.5, "rainfall": 1100, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "pomegranate", "N": 95, "P": 58, "K": 72, "temp": 26, "humidity": 60, "ph": 7.0, "rainfall": 700, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "sapota", "N": 105, "P": 62, "K": 78, "temp": 27, "humidity": 68, "ph": 6.5, "rainfall": 1000, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "pineapple", "N": 85, "P": 52, "K": 68, "temp": 26, "humidity": 72, "ph": 5.5, "rainfall": 1200, "type": "fruit", "season": "perennial", "days": 450},
    {"crop": "litchi", "N": 100, "P": 60, "K": 75, "temp": 25, "humidity": 70, "ph": 6.5, "rainfall": 1300, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "jackfruit", "N": 110, "P": 65, "K": 82, "temp": 27, "humidity": 75, "ph": 6.5, "rainfall": 1400, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "watermelon", "N": 80, "P": 50, "K": 60, "temp": 26, "humidity": 65, "ph": 6.5, "rainfall": 500, "type": "fruit", "season": "kharif", "days": 90},
    {"crop": "muskmelon", "N": 75, "P": 48, "K": 55, "temp": 25, "humidity": 64, "ph": 6.5, "rainfall": 450, "type": "fruit", "season": "kharif", "days": 85},
    {"crop": "strawberry", "N": 70, "P": 45, "K": 50, "temp": 20, "humidity": 68, "ph": 6.0, "rainfall": 600, "type": "fruit", "season": "rabi", "days": 120},
    
    # Spices (10 crops)
    {"crop": "ginger", "N": 100, "P": 60, "K": 80, "temp": 26, "humidity": 75, "ph": 6.0, "rainfall": 1500, "type": "spice", "season": "kharif", "days": 240},
    {"crop": "turmeric", "N": 110, "P": 65, "K": 85, "temp": 25, "humidity": 75, "ph": 6.5, "rainfall": 1400, "type": "spice", "season": "kharif", "days": 270},
    {"crop": "garlic", "N": 90, "P": 55, "K": 70, "temp": 22, "humidity": 60, "ph": 6.5, "rainfall": 450, "type": "spice", "season": "rabi", "days": 150},
    {"crop": "coriander", "N": 60, "P": 40, "K": 50, "temp": 23, "humidity": 62, "ph": 6.5, "rainfall": 400, "type": "spice", "season": "rabi", "days": 90},
    {"crop": "cumin", "N": 50, "P": 35, "K": 45, "temp": 24, "humidity": 55, "ph": 7.0, "rainfall": 350, "type": "spice", "season": "rabi", "days": 120},
    {"crop": "fenugreek", "N": 55, "P": 38, "K": 48, "temp": 22, "humidity": 58, "ph": 6.5, "rainfall": 380, "type": "spice", "season": "rabi", "days": 100},
    {"crop": "black_pepper", "N": 120, "P": 70, "K": 90, "temp": 26, "humidity": 80, "ph": 6.0, "rainfall": 2000, "type": "spice", "season": "perennial", "days": 365},
    {"crop": "cardamom", "N": 115, "P": 68, "K": 88, "temp": 22, "humidity": 78, "ph": 5.5, "rainfall": 2500, "type": "spice", "season": "perennial", "days": 365},
    {"crop": "chilli_pepper", "N": 95, "P": 58, "K": 65, "temp": 26, "humidity": 66, "ph": 6.5, "rainfall": 680, "type": "spice", "season": "kharif", "days": 115},
    {"crop": "fennel", "N": 65, "P": 42, "K": 52, "temp": 23, "humidity": 60, "ph": 6.8, "rainfall": 420, "type": "spice", "season": "rabi", "days": 110},
    
    # Oilseeds (10 crops)
    {"crop": "groundnut", "N": 25, "P": 60, "K": 70, "temp": 26, "humidity": 60, "ph": 6.5, "rainfall": 600, "type": "oilseed", "season": "kharif", "days": 110},
    {"crop": "soybean", "N": 30, "P": 65, "K": 75, "temp": 25, "humidity": 65, "ph": 6.5, "rainfall": 700, "type": "oilseed", "season": "kharif", "days": 100},
    {"crop": "mustard", "N": 80, "P": 50, "K": 60, "temp": 21, "humidity": 55, "ph": 6.5, "rainfall": 400, "type": "oilseed", "season": "rabi", "days": 120},
    {"crop": "sunflower", "N": 70, "P": 50, "K": 60, "temp": 24, "humidity": 58, "ph": 6.5, "rainfall": 500, "type": "oilseed", "season": "kharif", "days": 100},
    {"crop": "safflower", "N": 60, "P": 45, "K": 55, "temp": 23, "humidity": 52, "ph": 7.0, "rainfall": 450, "type": "oilseed", "season": "rabi", "days": 130},
    {"crop": "sesame", "N": 40, "P": 35, "K": 45, "temp": 26, "humidity": 55, "ph": 6.5, "rainfall": 500, "type": "oilseed", "season": "kharif", "days": 90},
    {"crop": "linseed", "N": 50, "P": 40, "K": 50, "temp": 20, "humidity": 54, "ph": 6.5, "rainfall": 450, "type": "oilseed", "season": "rabi", "days": 140},
    {"crop": "niger", "N": 35, "P": 30, "K": 40, "temp": 24, "humidity": 58, "ph": 6.5, "rainfall": 550, "type": "oilseed", "season": "kharif", "days": 120},
    {"crop": "castor", "N": 45, "P": 38, "K": 48, "temp": 25, "humidity": 56, "ph": 6.5, "rainfall": 600, "type": "oilseed", "season": "kharif", "days": 150},
    {"crop": "olive", "N": 85, "P": 52, "K": 68, "temp": 22, "humidity": 58, "ph": 7.0, "rainfall": 650, "type": "oilseed", "season": "perennial", "days": 365},
    
    # Cash Crops (5 crops)
    {"crop": "sugarcane", "N": 150, "P": 80, "K": 100, "temp": 27, "humidity": 75, "ph": 6.5, "rainfall": 1500, "type": "cash_crop", "season": "perennial", "days": 365},
    {"crop": "cotton", "N": 100, "P": 60, "K": 80, "temp": 26, "humidity": 60, "ph": 6.5, "rainfall": 700, "type": "cash_crop", "season": "kharif", "days": 160},
    {"crop": "tobacco", "N": 80, "P": 50, "K": 60, "temp": 24, "humidity": 65, "ph": 6.5, "rainfall": 600, "type": "cash_crop", "season": "rabi", "days": 120},
    {"crop": "jute", "N": 90, "P": 55, "K": 65, "temp": 27, "humidity": 80, "ph": 6.5, "rainfall": 1500, "type": "cash_crop", "season": "kharif", "days": 120},
    {"crop": "hemp", "N": 95, "P": 58, "K": 68, "temp": 23, "humidity": 65, "ph": 6.8, "rainfall": 700, "type": "cash_crop", "season": "kharif", "days": 110},
    
    # Plantation Crops (5 crops)
    {"crop": "tea", "N": 120, "P": 70, "K": 90, "temp": 23, "humidity": 80, "ph": 5.5, "rainfall": 2000, "type": "plantation", "season": "perennial", "days": 365},
    {"crop": "coffee", "N": 110, "P": 65, "K": 85, "temp": 24, "humidity": 75, "ph": 6.0, "rainfall": 1800, "type": "plantation", "season": "perennial", "days": 365},
    {"crop": "rubber", "N": 100, "P": 60, "K": 80, "temp": 27, "humidity": 80, "ph": 6.0, "rainfall": 2500, "type": "plantation", "season": "perennial", "days": 365},
    {"crop": "coconut", "N": 130, "P": 75, "K": 95, "temp": 28, "humidity": 75, "ph": 6.5, "rainfall": 1500, "type": "plantation", "season": "perennial", "days": 365},
    {"crop": "arecanut", "N": 125, "P": 72, "K": 92, "temp": 26, "humidity": 77, "ph": 6.5, "rainfall": 1800, "type": "plantation", "season": "perennial", "days": 365},
    
    # Additional Specialty Crops (5 crops to reach 100)
    {"crop": "custard_apple", "N": 98, "P": 60, "K": 75, "temp": 26, "humidity": 66, "ph": 6.5, "rainfall": 950, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "dragon_fruit", "N": 92, "P": 56, "K": 70, "temp": 27, "humidity": 68, "ph": 6.0, "rainfall": 800, "type": "fruit", "season": "perennial", "days": 365},
    {"crop": "cashew", "N": 115, "P": 68, "K": 88, "temp": 27, "humidity": 72, "ph": 6.0, "rainfall": 1300, "type": "nuts", "season": "perennial", "days": 365},
    {"crop": "almond", "N": 105, "P": 62, "K": 80, "temp": 22, "humidity": 55, "ph": 7.0, "rainfall": 600, "type": "nuts", "season": "perennial", "days": 365},
    {"crop": "walnut", "N": 110, "P": 65, "K": 82, "temp": 20, "humidity": 58, "ph": 6.5, "rainfall": 750, "type": "nuts", "season": "perennial", "days": 365},
]

# Create DataFrame
df = pd.DataFrame(crops_data)

# Rename temp to temperature
df.rename(columns={'temp': 'temperature'}, inplace=True)

# Add label column (for classification)
df['label'] = df['crop']

# Reorder columns to match expected format
df = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']]

# Save dataset
output_path = Path(__file__).parent / 'Crop_recommendation_100.csv'
df.to_csv(output_path, index=False)

print(f"✅ Generated 100-crop dataset: {output_path}")
print(f"📊 Total crops: {len(df)}")
print(f"🌾 Unique crops: {df['label'].nunique()}")
print(f"\n📋 Crop distribution by type:")
crops_extended = pd.DataFrame(crops_data)
print(crops_extended['type'].value_counts())

# Also create backup of original dataset
original_path = Path(__file__).parent / 'Crop_recommendation.csv'
backup_path = Path(__file__).parent / 'Crop_recommendation_46_backup.csv'
if original_path.exists():
    import shutil
    shutil.copy(original_path, backup_path)
    print(f"\n✅ Backed up original dataset to: {backup_path}")

# Replace original with new dataset
df.to_csv(original_path, index=False)
print(f"✅ Replaced {original_path} with 100-crop dataset")
