import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("indian_agriculture_ml_dataset.csv")
features = [
    "soil_n",
    "soil_p",
    "soil_k",
    "temperature_avg_c",
    "humidity_pct",
    "avg_rainfall_mm",
    "soil_ph",
]
# Add categorical if possible
if "soil_texture" in df.columns:
    df["soil_texture_enc"] = LabelEncoder().fit_transform(df["soil_texture"])
    features.append("soil_texture_enc")
if "agro_climatic_zone" in df.columns:
    df["agro_enc"] = LabelEncoder().fit_transform(df["agro_climatic_zone"])
    features.append("agro_enc")

target = "crop_name"
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

mi = mutual_info_classif(X, y)
print(dict(zip(features, mi)))
