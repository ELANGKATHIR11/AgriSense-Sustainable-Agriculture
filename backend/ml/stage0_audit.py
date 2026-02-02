import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------
# STAGE 0: DATA & SIGNAL AUDIT
# ----------------------------------------------------

DATA_PATH = Path("f:/AGRISENSEFULL-STACK/indian_agriculture_ml_dataset.csv")
ENHANCED_DIR = Path("f:/AGRISENSEFULL-STACK/backend/ml/datasets_enhanced")


def audit_dataset(df, name):
    print(f"\n--- Auditing {name} ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # 1. Water Signal (Physics)
    print("\n[Audit 1: Water Requirement]")
    if "et0_mm" in df.columns:
        # Check simple correlation: Area/ET0 vs Water
        # Physics: Water ~ Area * ET0
        # Check correlation of Water vs (Area * ET0)
        # Note: 'water_required_m3' might be noisy in skeleton, checking anyway
        if "water_required_m3" in df.columns:
            df["phys_proxy"] = df["area_m2"] * df["et0_mm"]
            corr = df[["water_required_m3", "phys_proxy"]].corr().iloc[0, 1]
            print(f"Water vs (Area*ET0) Corr: {corr:.4f}")
            status = "DETERMINISTIC" if corr > 0.9 else "NOISY/PROBABILISTIC"
            print(f"Signal Status: {status}")
        else:
            print("Target 'water_required_m3' missing - Cannot validate signal.")

    # 2. Season Signal (Climate)
    print("\n[Audit 2: Season Classification]")
    if "season" in df.columns and "avg_rainfall_mm" in df.columns:
        # Check if Season separates by Rainfall/Temp
        print("Season vs Rainfall/Temp (Group Means):")
        print(df.groupby("season")[["avg_rainfall_mm", "avg_temp_c"]].mean())
        # Check separability (ANOVA F-stat proxy or simple variance check)
        # Low variance between groups = Low signal
        status = "OBSERVABLE"  # Placeholder, user needs to see the stats

    # 3. Crop Signal (Partial Observability)
    print("\n[Audit 3: Crop Classification]")
    if "crop_name" in df.columns:
        # Check standard deviation of NPK across crops
        # If std(mean_N_per_crop) is high, crops are distinct.
        # If std(mean_N_per_crop) is low, crops are identical (noise).
        crop_means = df.groupby("crop_name")[["soil_n", "soil_p", "avg_temp_c"]].mean()
        print("Variance of Feature Means across Crops (Higher = Better):")
        print(crop_means.std())
        if crop_means["soil_n"].std() < 5:
            print("⚠️  WARNING: Crops have identical soil distributions (NOISE)")
        else:
            print("✅ Signal Detected: Crops have distinct soil profiles")

    # 4. Yield Signal (Weak Observability)
    print("\n[Audit 4: Yield Prediction]")
    if "yield_kg" in df.columns and "ndvi_mean" in df.columns:
        corr = df["yield_kg"].corr(df["ndvi_mean"])
        print(f"Yield vs NDVI Corr: {corr:.4f}")
        status = "OBSERVABLE" if abs(corr) > 0.3 else "WEAK/NOISE"
        print(f"Signal Status: {status}")


def main():
    # Audit 1: Original Skeleton
    try:
        df_orig = pd.read_csv(DATA_PATH)
        audit_dataset(df_orig, "ORIGINAL (indian_agriculture_ml_dataset.csv)")
    except Exception as e:
        print(f"Could not load Original: {e}")

    # Audit 2: Enhanced (if needed for comparison)
    print("\n" + "=" * 40)
    print("COMPARISON: ENHANCED GENERATED DATA")
    try:
        # Load one of the enhanced files to compare signal
        df_crop = pd.read_csv(
            ENHANCED_DIR / "crop_recommendation/crop_data_enhanced.csv"
        )
        # Rename for consistency if needed or just audit strictly
        # Enhance data has 'label' instead of 'crop_name', 'N' instead of 'soil_n'
        df_crop = df_crop.rename(
            columns={
                "label": "crop_name",
                "N": "soil_n",
                "P": "soil_p",
                "temperature": "avg_temp_c",
            }
        )

        print("\n--- Auditing Enhanced Crop Data ---")
        crop_means = df_crop.groupby("crop_name")[
            ["soil_n", "soil_p", "avg_temp_c"]
        ].mean()
        print("Variance of Feature Means across Crops:")
        print(crop_means.std())

    except Exception as e:
        print(f"Could not load Enhanced: {e}")


if __name__ == "__main__":
    main()
