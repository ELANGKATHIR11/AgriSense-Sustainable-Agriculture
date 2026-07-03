"""
AGRISENSE ML Inference Service — v2.0
Fixed: yield now uses N,P,K | irrigation uses crop encoding | disease uses real ML model.
"""
import joblib
import numpy as np
import os
import logging

logger = logging.getLogger("AgrisenseML")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "ml", "models")

_crop_model        = None
_crop_encoder      = None
_crop_feature_cols = None
_yield_model       = None
_yield_encoders    = None
_yield_feature_cols= None
_irrigation_bundle = None
_disease_risk      = None


def _load_crop():
    global _crop_model, _crop_encoder, _crop_feature_cols
    if _crop_model is None:
        _crop_model   = joblib.load(os.path.join(MODEL_DIR, "crop_recommendation_xgb.joblib"))
        _crop_encoder = joblib.load(os.path.join(MODEL_DIR, "crop_label_encoder.joblib"))
        try:
            _crop_feature_cols = joblib.load(os.path.join(MODEL_DIR, "crop_feature_cols.joblib"))
        except Exception:
            _crop_feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        logger.info("Crop recommendation model loaded")
    return _crop_model, _crop_encoder, _crop_feature_cols


def _load_yield():
    global _yield_model, _yield_encoders, _yield_feature_cols
    if _yield_model is None:
        _yield_model    = joblib.load(os.path.join(MODEL_DIR, "yield_prediction_catboost.joblib"))
        _yield_encoders = joblib.load(os.path.join(MODEL_DIR, "yield_encoders.joblib"))
        try:
            _yield_feature_cols = joblib.load(os.path.join(MODEL_DIR, "yield_feature_cols.joblib"))
        except Exception:
            _yield_feature_cols = ['crop_enc', 'season_enc', 'state_enc',
                                   'area_ha', 'rainfall_mm', 'fertilizer_kg', 'pesticide_kg']
        logger.info("Yield prediction model loaded")
    return _yield_model, _yield_encoders, _yield_feature_cols


def _load_irrigation():
    global _irrigation_bundle
    if _irrigation_bundle is None:
        bundle = joblib.load(os.path.join(MODEL_DIR, "irrigation_rf.joblib"))
        if isinstance(bundle, dict):
            _irrigation_bundle = bundle
        else:
            # Legacy single model
            _irrigation_bundle = {'lgb': bundle, 'gb': None, 'scaler': None,
                                   'feature_cols': ['moisture', 'temperature', 'humidity'],
                                   'blend': [1.0, 0.0]}
        logger.info("Irrigation model loaded")
    return _irrigation_bundle


def _load_disease_risk():
    global _disease_risk
    if _disease_risk is None:
        path = os.path.join(MODEL_DIR, "disease_risk_model.joblib")
        if os.path.exists(path):
            _disease_risk = joblib.load(path)
            logger.info("Disease risk model loaded")
    return _disease_risk


# ── Public inference functions ───────────────────────────────────────────────

def predict_crop(N: int, P: int, K: int, temperature: float,
                 humidity: float, ph: float, rainfall: float) -> dict:
    """Return top-3 crop recommendations with suitability scores."""
    try:
        model, encoder, feature_cols = _load_crop()
    except Exception as e:
        logger.error(f"Failed to load crop model: {e}")
        raise RuntimeError("Crop recommendation model not available. Please run ml/train_all_models.py")

    # Build feature dict supporting both old (7 cols) and new (13 cols) feature sets
    NPK_ratio  = (N + K) / (P + 1)
    NK_ratio   = N / (K + 1)
    ph_bin     = min(7, max(0, int((ph - 4.5) / 0.5)))
    temp_bin   = min(5, max(0, int((temperature - 10) / 5)))
    humid_cls  = int(humidity > 70)
    rain_cls   = min(4, max(0, int(rainfall / 75)))

    feat_dict = {
        'N': N, 'P': P, 'K': K,
        'temperature': temperature, 'humidity': humidity,
        'ph': ph, 'rainfall': rainfall,
        'NPK_ratio': NPK_ratio, 'NK_ratio': NK_ratio,
        'ph_bin': ph_bin, 'temp_bin': temp_bin,
        'humid_class': humid_cls, 'rain_class': rain_cls
    }

    features = np.array([[feat_dict.get(c, 0) for c in feature_cols]])

    probas     = model.predict_proba(features)[0]
    top_idx    = np.argsort(probas)[::-1][:3]

    CROP_DESCRIPTIONS = {
        "Rice":    "High rainfall crop, paddy fields, warm humid climate",
        "Wheat":   "Cool dry season grain, ideal for North Indian plains",
        "Maize":   "Versatile cereal, warm climate, moderate rainfall",
        "Sugarcane": "High water requirement, tropical/subtropical regions",
        "Cotton":  "Dryland crop, black soil, moderate rainfall",
        "Jute":    "High humidity, alluvial soil, heavy rainfall",
        "Coffee":  "Tropical highland crop, well-drained acidic soil",
        "Tea":     "Cool humid mountainous regions, acidic pH",
    }

    crops = []
    for idx in top_idx:
        name       = encoder.inverse_transform([idx])[0]
        suitability = round(float(probas[idx]) * 100, 1)
        crops.append({
            "name":             name,
            "suitability":      suitability,
            "description":      CROP_DESCRIPTIONS.get(name,
                f"Suitable for N={N}, P={P}, K={K} ppm soil, pH={ph}, {temperature}°C, {rainfall}mm rainfall."),
            "optimalConditions": f"Ensemble model (XGBoost+LightGBM+RandomForest) prediction."
        })

    # pH advisory
    if 6.0 <= ph <= 7.0:
        ph_status = "Healthy neutral optimal pH zone — ideal for most crops."
    elif ph < 6.0:
        ph_status = f"Acidic (pH={ph}). Apply agricultural lime to raise pH above 6.0."
    else:
        ph_status = f"Alkaline (pH={ph}). Apply elemental sulfur to lower pH below 7.5."

    nutrition = (
        f"Nitrogen ({N} ppm): {'adequate ✓' if N > 40 else 'low — apply urea or compost ⚠️'}. "
        f"Phosphorus ({P} ppm): {'adequate ✓' if P > 20 else 'deficient — apply DAP ⚠️'}. "
        f"Potassium ({K} ppm): {'adequate ✓' if K > 20 else 'low — apply MOP or SOP ⚠️'}."
    )

    return {"crops": crops, "optimalPH": ph_status, "nutritionStatus": nutrition}


def predict_irrigation(moisture: float, temperature: float,
                       humidity: float, crop_type: str = None) -> dict:
    """Predict water requirement using trained LightGBM+GB blend."""
    try:
        bundle = _load_irrigation()
    except Exception as e:
        logger.error(f"Failed to load irrigation model: {e}")
        raise RuntimeError("Irrigation model not available. Please run ml/train_all_models.py")

    lgb_model   = bundle['lgb']
    gb_model    = bundle.get('gb')
    scaler      = bundle.get('scaler')
    feature_cols= bundle.get('feature_cols', ['moisture', 'temperature', 'humidity'])
    blend       = bundle.get('blend', [0.6, 0.4])

    # Engineer features
    moisture_deficit = max(0, 45 - moisture)
    heat_stress      = float(temperature > 32)
    drought_stress   = float(moisture < 25)
    temp_humid_idx   = temperature * (1 - humidity / 100)

    feat_dict = {
        'moisture': moisture,
        'temperature': temperature,
        'humidity': humidity,
        'moisture_deficit': moisture_deficit,
        'heat_stress': heat_stress,
        'drought_stress': drought_stress,
        'temp_humid_idx': temp_humid_idx,
        'nitrogen': 45.0,
        'phosphorus': 38.0,
        'potassium': 42.0
    }

    features = np.array([[feat_dict.get(c, 0) for c in feature_cols]])

    if scaler is not None:
        features = scaler.transform(features)

    pred_lgb = lgb_model.predict(features)
    if gb_model is not None:
        pred_gb  = gb_model.predict(features)
        pred_raw = blend[0] * pred_lgb + blend[1] * pred_gb
    else:
        pred_raw = pred_lgb

    # If model was trained on log-transformed target, reverse-transform
    if bundle.get('log_transform', False):
        pred_raw = np.expm1(pred_raw)

    water_req = max(0, round(float(pred_raw[0] if hasattr(pred_raw, '__len__') else pred_raw)))

    CROP_KCS = {
        "Rice": 1.20, "Maize": 1.15, "Sugarcane": 1.25, "Wheat": 1.10,
        "Cotton": 1.05, "Tomato": 1.15, "Potato": 1.10, "Soybean": 1.10,
    }
    kc = CROP_KCS.get(crop_type or "", 1.0)
    water_req = round(water_req * kc)

    if water_req == 0:
        status   = "Adequate"
        duration = 0
        advice   = "Soil moisture within optimal range. No irrigation needed."
        schedule = "Standby — monitor daily moisture readings."
    elif moisture < 20:
        status   = "CRITICAL — Severe Drought Stress"
        duration = round(water_req / 35)
        advice   = f"IMMEDIATE irrigation required for {crop_type or 'crops'}. Activate full drip network."
        schedule = "Every 4 hours for 48 hours, then reassess."
    elif moisture < 30:
        status   = "Low Moisture — Irrigation Required"
        duration = round(water_req / 42)
        advice   = f"Schedule drip irrigation for {crop_type or 'crops'} to prevent wilting."
        schedule = "Dawn and dusk cycles, 3 days."
    else:
        status   = "Moderate Moisture Stress"
        duration = round(water_req / 50)
        advice   = f"Preventative micro-drip for {crop_type or 'crops'} recommended."
        schedule = "Morning irrigation only."

    return {
        "waterRequiredLiters": water_req,
        "moistureStatus":      status,
        "advice":              advice,
        "durationMinutes":     duration,
        "irrigationSchedule":  schedule
    }


def predict_yield(area_acres: float, avg_rainfall: float,
                  avg_temp: float, crop_type: str,
                  nitrogen: float = 45.0,
                  phosphorus: float = 38.0,
                  potassium: float = 42.0) -> dict:
    """Predict yield using trained CatBoost model with full feature set."""
    try:
        model, encoders, feature_cols = _load_yield()
    except Exception as e:
        logger.error(f"Failed to load yield model: {e}")
        raise RuntimeError("Yield model not available. Please run ml/train_all_models.py")

    area_ha = area_acres * 0.4047  # acres → hectares

    # Encode
    crop_enc   = encoders.get("Crop")
    season_enc = encoders.get("Season")
    state_enc  = encoders.get("State")

    def safe_encode(enc, val, default=0):
        if enc is None: return default
        try:
            return enc.transform([str(val).strip()])[0]
        except (ValueError, KeyError):
            return len(enc.classes_) // 2

    crop_encoded   = safe_encode(crop_enc, crop_type, 0)
    season_encoded = safe_encode(season_enc, "Kharif", 0)
    state_encoded  = safe_encode(state_enc, "Karnataka", 0)

    # Derived features (matching training feature set)
    fertilizer_kg = (nitrogen + phosphorus + potassium) * area_ha * 0.15
    pesticide_kg  = area_ha * 0.8
    npk_total     = nitrogen + phosphorus + potassium
    rain_per_ha   = avg_rainfall / (area_ha + 1)
    fert_per_ha   = fertilizer_kg / (area_ha + 1)

    feat_dict = {
        'crop_enc':    crop_encoded,
        'season_enc':  season_encoded,
        'state_enc':   state_encoded,
        'area_ha':     area_ha,
        'rainfall_mm': avg_rainfall,
        'fertilizer_kg': fertilizer_kg,
        'pesticide_kg':  pesticide_kg,
        'nitrogen':    nitrogen,
        'phosphorus':  phosphorus,
        'potassium':   potassium,
        'npk_total':   npk_total,
        'rain_per_ha': rain_per_ha,
        'fert_per_ha': fert_per_ha,
        # Legacy features (fallback)
        'Area':        area_ha,
        'Annual_Rainfall': avg_rainfall,
        'Fertilizer':  fertilizer_kg,
        'Pesticide':   pesticide_kg,
        'Crop':        crop_encoded,
        'Season':      season_encoded,
        'State':       state_encoded,
    }

    features = np.array([[feat_dict.get(c, 0) for c in feature_cols]])
    predicted_yield_per_ha = float(model.predict(features)[0])
    predicted_yield_per_ha = max(0.05, predicted_yield_per_ha)

    # Convert from t/ha to total tons
    total_yield = round(predicted_yield_per_ha * area_ha, 2)
    conf_min    = round(total_yield * 0.88, 2)
    conf_max    = round(total_yield * 1.12, 2)

    # Updated market prices (INR per quintal → ₹/ton ×10)
    MARKET_PRICES = {
        "Rice": 3100, "Wheat": 2850, "Maize": 2200, "Cotton": 6500,
        "Sugarcane": 3500, "Soybean": 4400, "Tomato": 1200, "Potato": 1500,
        "Onion": 2500, "Groundnut": 6000, "Sunflower": 5800, "Mustard": 5400,
        "Jute": 4500, "Tea": 22000, "Coffee": 18000, "Turmeric": 14000,
    }
    price_per_ton = MARKET_PRICES.get(crop_type, 3500)

    return {
        "predictedYieldTons":    total_yield,
        "confidenceMin":         conf_min,
        "confidenceMax":         conf_max,
        "marketValueEstimate":   round(total_yield * price_per_ton),
        "yieldBreakdown": (
            f"CatBoost YieldPredictor: {predicted_yield_per_ha:.2f} t/ha × {area_ha:.1f} ha = {total_yield:.2f} t total. "
            f"NPK input: N={nitrogen}, P={phosphorus}, K={potassium} ppm. "
            f"Rainfall: {avg_rainfall}mm, Temp: {avg_temp}°C."
        )
    }


def predict_disease_risk(humidity: float, temperature: float,
                          leaf_wetness_hours: float = 8.0,
                          soil_moisture: float = 40.0,
                          rainfall_mm: float = 5.0) -> dict:
    """Predict disease risk level from environmental conditions."""
    bundle = _load_disease_risk()

    if bundle is None:
        # Fallback heuristic while model is training
        weather_intensity = (humidity / 100.0) * (temperature / 25.0)
        pathogen_risk = min(98, round(weather_intensity * 45 + (25 if soil_moisture > 55 else 0)))
        level = "high" if pathogen_risk > 60 else ("medium" if pathogen_risk > 30 else "low")
        return {"riskScore": pathogen_risk, "riskLevel": level,
                "outbreakProbability": round(pathogen_risk * 0.9),
                "environmentalPropensity": round(weather_intensity * 100)}

    model   = bundle['model']
    scaler  = bundle['scaler']
    f_cols  = bundle['feature_cols']
    classes = bundle.get('classes', ['low', 'medium', 'high'])

    feat_dict = {
        'humidity': humidity,
        'temperature': temperature,
        'leaf_wetness_hours': leaf_wetness_hours,
        'soil_moisture_pct': soil_moisture,
        'rainfall_mm': rainfall_mm,
        'temperature_c': temperature,
        'humidity_pct': humidity,
    }
    features = np.array([[feat_dict.get(c, 0) for c in f_cols]])

    if scaler is not None:
        features = scaler.transform(features)

    proba = model.predict_proba(features)[0]
    pred  = int(np.argmax(proba))
    risk_score = round(proba[2] * 100 if len(proba) >= 3 else proba[-1] * 100)

    return {
        "riskScore":              risk_score,
        "riskLevel":              classes[pred],
        "outbreakProbability":    round(risk_score * 0.88),
        "environmentalPropensity": round(max(proba) * 100),
        "classProbabilities":     {classes[i]: round(p*100, 1) for i, p in enumerate(proba)}
    }


_fert_model = None
_fert_encoder = None
_fert_feature_cols = None

def _load_fertilizer():
    global _fert_model, _fert_encoder, _fert_feature_cols
    if _fert_model is None:
        _fert_model = joblib.load(os.path.join(MODEL_DIR, "fertilizer_recommendation_catboost.joblib"))
        _fert_encoder = joblib.load(os.path.join(MODEL_DIR, "fertilizer_label_encoder.joblib"))
        try:
            _fert_feature_cols = joblib.load(os.path.join(MODEL_DIR, "fertilizer_feature_cols.joblib"))
        except Exception:
            _fert_feature_cols = ['temperature', 'humidity', 'moisture', 'soil_type', 'crop_type', 'nitrogen', 'potassium', 'phosphorus']
        logger.info("Fertilizer recommendation model loaded")
    return _fert_model, _fert_encoder, _fert_feature_cols

def predict_fertilizer(temperature: float, humidity: float, moisture: float,
                       soil_type: str, crop_type: str,
                       nitrogen: float, potassium: float, phosphorus: float) -> dict:
    """Predict top fertilizer recommendation using CatBoost."""
    import pandas as pd
    try:
        model, encoder, feature_cols = _load_fertilizer()
    except Exception as e:
        logger.error(f"Failed to load fertilizer model: {e}")
        raise RuntimeError("Fertilizer model not available. Please run ml/train_all_models.py")

    feat_dict = {
        'temperature': temperature,
        'humidity': humidity,
        'moisture': moisture,
        'soil_type': str(soil_type).strip(),
        'crop_type': str(crop_type).strip(),
        'nitrogen': nitrogen,
        'potassium': potassium,
        'phosphorus': phosphorus
    }

    df_input = pd.DataFrame([[feat_dict.get(c) for c in feature_cols]], columns=feature_cols)
    probas = model.predict_proba(df_input)[0]
    best_idx = np.argmax(probas)
    
    recommended_name = str(encoder.inverse_transform([best_idx])[0])
    confidence = float(probas[best_idx])

    # Dynamic advisories based on recommended fertilizer type
    ADVISORIES = {
        "Urea": "High Nitrogen booster. Apply split-doses around leaf vegetative stages. Avoid directly touching plant stalks.",
        "DAP": "Diammonium Phosphate. High Phosphorus content. Ideal for root establishment during sowing or early vegetative phase.",
        "MOP": "Muriate of Potash. Potassium booster. Promotes drought resistance, grain filling, and disease tolerance.",
        "10-26-26": "Balanced NPK mixture, high in P and K. Promotes crop flowering, root branching, and fruit formation.",
        "14-35-14": "High phosphorous mixture. Excellent starter fertilizer for tuber crops, maize, and oilseeds.",
        "20-20": "Equal Nitrogen and Phosphorus blend. Great for leafy green vegetative development and structural vigor.",
        "28-28": "High concentrated nitrogen-phosphorus blend. Best suited for high-density cereal farming."
    }

    return {
        "recommendedFertilizer": recommended_name,
        "confidence": confidence,
        "advisory": ADVISORIES.get(recommended_name, f"Apply recommended {recommended_name} according to standard soil dosage instructions.")
    }


# ── DIGITAL TWIN RESIDUAL CORRECTED PREDICTIONS ──────────────────────────────
_twin_water_model = None
_twin_disease_model = None
_twin_growth_model = None
_twin_yield_model = None

def _load_twin_models():
    global _twin_water_model, _twin_disease_model, _twin_growth_model, _twin_yield_model
    if _twin_water_model is None:
        _twin_water_model = joblib.load(os.path.join(MODEL_DIR, "twin_water_stress_index_catboost.joblib"))
        _twin_disease_model = joblib.load(os.path.join(MODEL_DIR, "twin_disease_risk_index_catboost.joblib"))
        _twin_growth_model = joblib.load(os.path.join(MODEL_DIR, "twin_growth_simulation_index_catboost.joblib"))
        _twin_yield_model = joblib.load(os.path.join(MODEL_DIR, "twin_yield_forecast_catboost.joblib"))
    return _twin_water_model, _twin_disease_model, _twin_growth_model, _twin_yield_model

def _physics_water_stress(row):
    moisture = row.get('Soil_Moisture_Surface', 35.0)
    temp = row.get('Air_Temperature', 25.0)
    base_stress = np.clip((45.0 - moisture) / 45.0, 0.0, 1.0)
    heat_factor = 1.0 + np.maximum(0.0, temp - 32.0) * 0.05
    return np.clip(base_stress * heat_factor, 0.0, 1.0)

def _physics_disease_risk(row):
    humidity = row.get('Humidity', 60.0)
    temp = row.get('Air_Temperature', 25.0)
    rainfall_24 = row.get('Rainfall_24h', 5.0)
    risk = (humidity / 100.0) * (1.0 - np.abs(temp - 23.0) / 10.0) * (1.0 + rainfall_24 / 10.0)
    return np.clip(risk, 0.0, 1.0)

def _physics_growth_index(row):
    temp = row.get('Air_Temperature', 25.0)
    moisture = row.get('Soil_Moisture_Surface', 35.0)
    age = row.get('Plant_Age', 30.0)
    gdd = np.maximum(0.0, temp - 10.0)
    index = (gdd / 15.0) * (moisture / 35.0) * (1.0 + age / 120.0)
    return np.clip(index, 0.0, 1.0)

def predict_digital_twin_state(nitrogen, phosphorus, potassium, air_temp, humidity, pH, moisture, rainfall=0.0):
    """
    Predicts physics-guided hybrid digital twin state index variables.
    Computes baseline agronomic equations first, and then adds CatBoost predicted residuals.
    """
    w_mod, d_mod, g_mod, y_mod = _load_twin_models()
    
    # Map raw features to expected model schema inputs
    row_dict = {
        'Nitrogen': nitrogen,
        'Phosphorus': phosphorus,
        'Potassium': potassium,
        'Air_Temperature': air_temp,
        'Humidity': humidity,
        'Soil_pH': pH,
        'Soil_Moisture_Surface': moisture,
        'Rainfall': rainfall
    }
    
    # 1. Compute Physics baselines
    water_stress_phys = _physics_water_stress(row_dict)
    disease_risk_phys = _physics_disease_risk(row_dict)
    growth_sim_phys = _physics_growth_index(row_dict)
    
    # Helper to clean feature vectors & predict residual
    def predict_residual(bundle):
        f_cols = bundle['feature_cols']
        feats = np.array([[row_dict.get(c, 0.0) for c in f_cols]])
        if bundle['scaler'] is not None:
            feats = bundle['scaler'].transform(feats)
        return float(bundle['model'].predict(feats)[0])
        
    water_stress_res = predict_residual(w_mod)
    disease_risk_res = predict_residual(d_mod)
    growth_sim_res = predict_residual(g_mod)
    
    # 2. Predict Yield Forecast directly
    feats_yf = np.array([[row_dict.get(c, 0.0) for c in y_mod['feature_cols']]])
    if y_mod['scaler'] is not None:
        feats_yf = y_mod['scaler'].transform(feats_yf)
    yield_forecast = float(y_mod['model'].predict(feats_yf)[0])
    
    # 3. Sum baselines and residuals
    water_stress = float(np.clip(water_stress_phys + water_stress_res, 0.0, 1.0))
    disease_risk = float(np.clip(disease_risk_phys + disease_risk_res, 0.0, 1.0))
    growth_sim = float(np.clip(growth_sim_phys + growth_sim_res, 0.0, 1.0))
    
    return {
        "waterStressIndex": round(water_stress, 4),
        "diseaseRiskIndex": round(disease_risk, 4),
        "growthSimulationIndex": round(growth_sim, 4),
        "yieldForecast": round(yield_forecast, 4),
        "physicsBaselines": {
            "waterStressBaseline": round(water_stress_phys, 4),
            "diseaseRiskBaseline": round(disease_risk_phys, 4),
            "growthSimulationBaseline": round(growth_sim_phys, 4)
        }
    }


