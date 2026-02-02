import pandas as pd
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('indian_agriculture_ml_dataset.csv')
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Basic info
print("\n--- Basic Info ---")
print(df.info())

# Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Columns
print("\n--- Columns ---")
print(df.columns.tolist())

# Target variable distribution (assuming 'yield_kg' or 'crop_name' might be targets depending on the model)
if 'yield_kg' in df.columns:
    print("\n--- Yield Distribution ---")
    print(df['yield_kg'].describe())

if 'crop_name' in df.columns:
    print("\n--- Crop Name Distribution ---")
    print(df['crop_name'].value_counts())

# Check for categorical variables
print("\n--- Categorical Variables ---")
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"\n{col} unique values: {df[col].nunique()}")
    if df[col].nunique() < 20:
        print(df[col].value_counts())
