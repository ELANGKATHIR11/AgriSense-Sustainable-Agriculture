# Tabular Data Directory

Contains generated CSV datasets for crop recommendation and yield prediction models.

## Files:
- `india_crops_complete.csv` - Main crop recommendation dataset (22 crops × 2000 samples)
- `historical_yields.csv` - Yield prediction dataset with temporal features

## Schema:
### india_crops_complete.csv
| Column | Type | Description |
|--------|------|-------------|
| N | float | Nitrogen content (kg/ha) |
| P | float | Phosphorus content (kg/ha) |
| K | float | Potassium content (kg/ha) |
| temperature | float | Temperature (°C) |
| humidity | float | Relative humidity (%) |
| ph | float | Soil pH |
| rainfall | float | Annual rainfall (mm) |
| soil_type | str | Soil classification |
| label | str | Crop name |

### historical_yields.csv
Additional columns: year, pest_incidence, fertilizer_usage_kg, yield_t_ha
