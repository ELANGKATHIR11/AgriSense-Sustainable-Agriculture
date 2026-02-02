"""Preprocess dataset and export per-model feature files and scalers/encoders.

Usage:
    python backend/ml/preprocess_and_export.py

Creates files under `backend/ml/models/` and `backend/ml/processed`.
"""

# pyright: reportMissingTypeStubs=false
from pathlib import Path

import joblib  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.preprocessing import (LabelEncoder,  # type: ignore
                                   OrdinalEncoder, StandardScaler)

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
PROCESSED_DIR = BASE / "processed"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def ndvi_features(row):
    keys = [f"ndvi_t{i}" for i in range(1, 7)]
    vals = [row.get(k) for k in keys]
    vals = [v for v in vals if pd.notna(v)]
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if len(vals) >= 2:
        slope = float(np.polyfit(range(len(vals)), vals, 1)[0])
    else:
        slope = 0.0
    mx = float(np.max(vals))
    mn = float(np.min(vals))
    rng = mx - mn
    med = float(np.median(vals))
    return mean, std, slope, mx, mn, rng, med


def preprocess_and_export(csv_path: Path):
    df = load_data(csv_path)

    # standardize column names
    df = df.rename(
        columns={
            "soil_n": "N",
            "soil_p": "P",
            "soil_k": "K",
            "temperature_avg_c": "temperature",
            "humidity_pct": "humidity",
            "soil_ph": "ph",
            "rainfall_effective_mm": "rainfall",
        }
    )

    # ndvi features (mean, std, slope, max, min, range, median)
    ndvi_cols = [
        "ndvi_seq_mean",
        "ndvi_seq_std",
        "ndvi_seq_slope",
        "ndvi_seq_max",
        "ndvi_seq_min",
        "ndvi_seq_range",
        "ndvi_seq_median",
    ]
    ndvi_series = df.apply(lambda r: pd.Series(ndvi_features(r)), axis=1)
    df[ndvi_cols] = ndvi_series

    # compute EVI if spectral bands exist (nir/red/blue time series)
    def compute_evi_row(r):
        nir_keys = [f"nir_t{i}" for i in range(1, 7)]
        red_keys = [f"red_t{i}" for i in range(1, 7)]
        blue_keys = [f"blue_t{i}" for i in range(1, 7)]
        if all(k in r.index for k in nir_keys + red_keys + blue_keys):
            evis = []
            for i in range(6):
                nir = r.get(nir_keys[i])
                red = r.get(red_keys[i])
                blue = r.get(blue_keys[i])
                if pd.isna(nir) or pd.isna(red) or pd.isna(blue):
                    continue
                # EVI formula
                denom = nir + 6 * red - 7.5 * blue + 1
                if denom == 0:
                    continue
                evi = 2.5 * (nir - red) / denom
                evis.append(evi)
            if len(evis) == 0:
                return [0.0] * 7
            mean = float(np.mean(evis))
            std = float(np.std(evis))
            if len(evis) >= 2:
                slope = float(np.polyfit(range(len(evis)), evis, 1)[0])
            else:
                slope = 0.0
            mx = float(np.max(evis))
            mn = float(np.min(evis))
            rng = mx - mn
            med = float(np.median(evis))
            return [mean, std, slope, mx, mn, rng, med]
        return [0.0] * 7

    evi_cols = [
        "evi_seq_mean",
        "evi_seq_std",
        "evi_seq_slope",
        "evi_seq_max",
        "evi_seq_min",
        "evi_seq_range",
        "evi_seq_median",
    ]
    df[evi_cols] = df.apply(lambda r: pd.Series(compute_evi_row(r)), axis=1)

    # interaction terms
    inter_cols = {}
    if "N" in df.columns and "P" in df.columns:
        inter_cols["N_x_P"] = df["N"] * df["P"]
    if "N" in df.columns and "K" in df.columns:
        inter_cols["N_x_K"] = df["N"] * df["K"]
    if "temperature" in df.columns and "rainfall" in df.columns:
        inter_cols["temp_x_rain"] = df["temperature"] * df["rainfall"]
    if "ndvi_seq_mean" in df.columns and "rainfall" in df.columns:
        inter_cols["ndvi_x_rain"] = df["ndvi_seq_mean"] * df["rainfall"]
    if "evi_seq_mean" in df.columns and "rainfall" in df.columns:
        inter_cols["evi_x_rain"] = df["evi_seq_mean"] * df["rainfall"]
    for k, v in inter_cols.items():
        df[k] = v

    # simple numeric imputation: fill medians for numeric columns used
    numeric_cols = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "water_required_m3",
        "water_requirement",
        "yield_kg",
    ]
    present_numeric = [c for c in numeric_cols if c in df.columns]
    imputer = SimpleImputer(strategy="median")
    if present_numeric:
        df[present_numeric] = imputer.fit_transform(df[present_numeric])
        joblib.dump(imputer, MODELS_DIR / "preprocess_numeric_imputer.pkl")

    # categorical encoding for features (if present)
    cat_cols = [
        c
        for c in ["soil_texture", "growth_stage", "irrigation_method"]
        if c in df.columns
    ]
    if cat_cols:
        oenc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df[cat_cols] = oenc.fit_transform(df[cat_cols].fillna("__MISSING__"))
        cat_enc_path = MODELS_DIR / "preprocess_categorical_encoder.pkl"
        joblib.dump(oenc, cat_enc_path)

    # label encoders for targets used by ml_service
    targets = {
        "recommended_crop": "enhanced_crop_recommendation_encoder.pkl",
        "crop_type": "enhanced_crop_type_classification_encoder.pkl",
        "season": "enhanced_season_classification_encoder.pkl",
    }
    for col, fname in targets.items():
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna("__MISSING__").astype(str)
            le.fit(df[col])
            lab_path = MODELS_DIR / fname
            joblib.dump(le, lab_path)

    # Prepare per-model feature sets and fit scalers
    model_features = {
        "crop_recommendation": [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ],
        "yield_prediction": [
            "N",
            "P",
            "K",
            "temperature",
            "rainfall",
            "water_requirement",
            "growth_duration",
        ],
        "crop_type": [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ],
        "water_requirement": [
            "temperature",
            "humidity",
            "rainfall",
            "growth_duration",
        ],
        "season_classification": ["temperature", "rainfall", "humidity"],
    }

    for model_name, feats in model_features.items():
        feats_present = [f for f in feats if f in df.columns]
        if not feats_present:
            continue
        X = df[feats_present].astype(float).to_numpy()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        scaler_path = MODELS_DIR / f"enhanced_{model_name}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        # save compact CSV for sample/training use
        out_df = pd.DataFrame(Xs, columns=feats_present)
        out_path = PROCESSED_DIR / f"features_{model_name}.csv"
        out_df.to_csv(out_path, index=False)

    # Save processed full CSV (compact)
    df.to_csv(PROCESSED_DIR / "processed_ml_input.csv", index=False)
    print("Preprocessing complete. Artifacts saved to:")
    print(" -", MODELS_DIR)
    print(" -", PROCESSED_DIR)


if __name__ == "__main__":
    csv_path = (
        Path(__file__).parent.parent.parent
        / "indian_agriculture_ml_dataset.csv"
    )
    preprocess_and_export(csv_path)
