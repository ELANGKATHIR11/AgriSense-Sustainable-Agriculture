"""
Dataset validation script for `indian_agriculture_ml_dataset.csv`.
Generates: validation_report.json and flagged_rows.csv in the same folder.
"""
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
else:
    try:
        import pandas as pd  # type: ignore[import]
    except Exception:
        print("Missing dependency: pandas. Install with: pip install pandas")
        raise

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / 'indian_agriculture_ml_dataset.csv'
OUT_DIR = Path(__file__).resolve().parent
REPORT_JSON = OUT_DIR / 'validation_report.json'
FLAGGED_CSV = OUT_DIR / 'flagged_rows.csv'

if not CSV_PATH.exists():
    print(f"Dataset not found at {CSV_PATH}")
    sys.exit(1)

print(f"Reading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

# Columns of interest
soil_cols = [c for c in df.columns if c.startswith('soil_')]
ndvi_cols = [c for c in df.columns if c.startswith('ndvi') or c.startswith('evi') or c.startswith('ndwi')]
target_cols = ['yield_kg', 'water_required_m3']
geo_cols = ['latitude', 'longitude']
other_cols = ['field_id', 'sowing_day_of_year', 'season', 'area_m2']

report: dict[str, Any] = {}

# Missing value summary
report['missing'] = {}
for col in soil_cols + ndvi_cols + target_cols:
    if col in df.columns:
        miss = df[col].isna().sum()
        pct = 100.0 * miss / len(df)
        report['missing'][col] = {'missing_count': int(miss), 'missing_pct': round(pct, 3)}

# Numeric summaries
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
report['numeric_summary'] = {}
for c in num_cols:
    ser = pd.to_numeric(df[c], errors='coerce')
    report['numeric_summary'][c] = {
        'count': int(ser.count()),
        'mean': None if ser.count()==0 else float(ser.mean()),
        'std': None if ser.count()==0 else float(ser.std()),
        'min': None if ser.count()==0 else float(ser.min()),
        '25%': None if ser.count()==0 else float(ser.quantile(0.25)),
        '50%': None if ser.count()==0 else float(ser.median()),
        '75%': None if ser.count()==0 else float(ser.quantile(0.75)),
        'max': None if ser.count()==0 else float(ser.max())
    }

# Geo checks
bad_geo = pd.Series(False, index=df.index)
if 'latitude' in df.columns:
    bad_geo = bad_geo | (~df['latitude'].between(-90, 90))
if 'longitude' in df.columns:
    bad_geo = bad_geo | (~df['longitude'].between(-180, 180))
report['geo_issues'] = {'bad_geo_count': int(bad_geo.sum())}

# Duplicates by field_id
dupes = []
if 'field_id' in df.columns:
    dupmask = df.duplicated(subset=['field_id'], keep=False)
    dup_count = int(dupmask.sum())
    dup_sample = df.loc[dupmask, 'field_id'].drop_duplicates().tolist()
    report['duplicates_by_field_id'] = {'duplicate_rows': dup_count, 'unique_duplicate_field_ids_sample': dup_sample[:20]}
else:
    report['duplicates_by_field_id'] = {'duplicate_rows': 0, 'unique_duplicate_field_ids_sample': []}

# Units sanity checks
report['unit_checks'] = {}
# rainfall mm columns
for col in ['avg_rainfall_mm', 'rainfall_mm', 'rainfall_30d_mm', 'rainfall_60d_mm', 'rainfall_effective_mm']:
    if col in df.columns:
        ser = pd.to_numeric(df[col], errors='coerce')
        report['unit_checks'][col] = {
            'min': None if ser.count()==0 else float(ser.min()),
            'max': None if ser.count()==0 else float(ser.max()),
            'negative_count': int((ser < 0).sum())
        }
# water_required_m3
for col in ['water_required_m3']:
    if col in df.columns:
        ser = pd.to_numeric(df[col], errors='coerce')
        report['unit_checks'][col] = {
            'min': None if ser.count()==0 else float(ser.min()),
            'max': None if ser.count()==0 else float(ser.max()),
            'zeros_count': int((ser == 0).sum())
        }
# yield
for col in ['yield_kg']:
    if col in df.columns:
        ser = pd.to_numeric(df[col], errors='coerce')
        report['unit_checks'][col] = {
            'min': None if ser.count()==0 else float(ser.min()),
            'max': None if ser.count()==0 else float(ser.max()),
            'negative_count': int((ser < 0).sum())
        }

# Sowing day vs season heuristic: compute median sowing day per season and flag outliers > 90 days
flag_sowing = pd.Series(False, index=df.index)
if 'sowing_day_of_year' in df.columns and 'season' in df.columns:
    df['sowing_day_of_year_num'] = pd.to_numeric(df['sowing_day_of_year'], errors='coerce')
    season_groups = df.groupby('season')['sowing_day_of_year_num'].median().to_dict()
    report['season_sowing_median'] = {k: (None if pd.isna(v) else int(v)) for k, v in season_groups.items()}
    for season, med in season_groups.items():
        if pd.isna(med):
            continue
        mask = (df['season'] == season) & (df['sowing_day_of_year_num'].notna()) & (df['sowing_day_of_year_num'].sub(med).abs() > 90)
        flag_sowing = flag_sowing | mask
    report['season_sowing_flagged_count'] = int(flag_sowing.sum())
else:
    report['season_sowing_median'] = {}
    report['season_sowing_flagged_count'] = 0

# Collect flagged rows (union of issues)
flagged = pd.DataFrame()
flags = pd.Series(False, index=df.index)
# missing in key cols
for col in soil_cols + ndvi_cols + ['yield_kg']:
    if col in df.columns:
        flags = flags | df[col].isna()
# geo
flags = flags | bad_geo
# dupes
if 'field_id' in df.columns:
    flags = flags | df.duplicated(subset=['field_id'], keep=False)
# sowing misaligned
flags = flags | flag_sowing

flagged = df.loc[flags]
report['flagged_count'] = int(len(flagged))
report['rows'] = len(df)

# Save sample of flagged rows
if len(flagged) > 0:
    flagged_sample = flagged.sample(min(200, len(flagged))).to_dict(orient='records')
else:
    flagged_sample = []
report['flagged_sample'] = flagged_sample

# Write outputs
with open(REPORT_JSON, 'w', encoding='utf8') as f:
    json.dump(report, f, indent=2)

flagged.to_csv(FLAGGED_CSV, index=False)

print('\nValidation complete.')
print(f"Report: {REPORT_JSON}")
print(f"Flagged rows CSV: {FLAGGED_CSV} (rows: {len(flagged)})")
