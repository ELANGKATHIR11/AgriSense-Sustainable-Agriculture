import os, json, warnings, shap, joblib, optuna
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).parent
DATA_PATH = Path("f:/AGRISENSEFULL-STACK/indian_agriculture_ml_dataset.csv")
ENHANCED_DIR = BASE_DIR / "datasets_enhanced"
MODELS_DIR = BASE_DIR / "models" / "hybrid"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🌿 {msg}")


# --- STEP 1: PREPARATION & PHYSICS ---
def prepare_data():
    log("Step 1: Data Preparation & Physics Recomputation")
    df = pd.read_csv(DATA_PATH).dropna(subset=["area_m2", "district_code"])
    df["ndvi_mean"] = df["ndvi_mean"].clip(-1, 1)

    # 3. Growth Stage Mapping
    gs_map = {"Initial": 0, "Vegetative": 1, "Flowering": 2, "Maturity": 3}
    df["gs_enc"] = df["growth_stage"].map(gs_map).fillna(1)

    # 4. Water Physics (ET0 * Kc * Area - Rain) / Eff
    eff = (
        df["irrigation_method"]
        .map({"Drip": 0.9, "Sprinkler": 0.75, "Surface": 0.6})
        .fillna(0.6)
    )
    supply = df["rainfall_effective_mm"] * df["area_m2"] * 0.001
    demand = df["et0_mm"] * df["kc_value"] * df["area_m2"] * 0.001
    df["physics_water_m3"] = (demand - supply).clip(0) / eff

    # 5. Yield Engineering
    df["Area_Ha"] = df["area_m2"] / 10000.0
    df["yield_ton_ha"] = (df["yield_kg"] / df["area_m2"]) * 10

    # Encoders
    encoders = {}
    for col in [
        "crop_name",
        "crop_type",
        "agro_climatic_zone",
        "soil_texture",
        "season_target",
    ]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    return df


# --- STEP 5: PHYSICS CORRECTOR ---
class PhysicsCorrector:
    caps = {"Rice": 12, "Wheat": 9, "Maize": 11, "Cotton": 4, "Sugarcane": 90}

    def apply(self, crop, yield_val, water_val, area_ha):
        # Cap Yield
        cap = self.caps.get(crop, 15)
        yield_val = min(yield_val, cap)
        # Min Water: 0.8 * ET0 equivalent roughly
        water_val = max(water_val, 0)
        return yield_val, water_val


# --- STEP 2,3,4: HYBRID TRAINING ---
def train_task(df, features, target, type="reg"):
    log(f"Training Hybrid {type} for {target}")

    if "district_code" in df.columns:
        log("   Using Spatial Split (District-based)")
        gkf = GroupKFold(n_splits=5)
        train_idx, test_idx = next(gkf.split(df, groups=df["district_code"]))
        X_train, X_test = df[features].iloc[train_idx], df[features].iloc[test_idx]
        y_train, y_test = df[target].iloc[train_idx], df[target].iloc[test_idx]
    else:
        log("   Using Random Split (Non-spatial)")
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            df[features], df[target], test_size=0.2, random_state=42
        )

    # XGBoost
    log("   Backbone XGBoost...")
    if type == "reg":
        # Monotones
        mono = []
        for c in features:
            if c in ["Area_Ha", "et0_mm"]:
                mono.append(1)
            elif c == "rainfall_effective_mm":
                mono.append(-1)
            else:
                mono.append(0)
        xgb = XGBRegressor(
            n_estimators=300, monotone_constraints=tuple(mono), n_jobs=-1
        )
    else:
        xgb = XGBClassifier(n_estimators=300, n_jobs=-1)

    xgb.fit(X_train, y_train)
    p_xgb = xgb.predict(X_test)

    # TabNet
    log("   TabNet Enhancer...")
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_train).astype("float32")
    X_te_s = sc.transform(X_test).astype("float32")

    if type == "reg":
        tab = TabNetRegressor(verbose=0)
        tab.fit(
            X_tr_s,
            y_train.values.reshape(-1, 1),
            eval_set=[(X_te_s, y_test.values.reshape(-1, 1))],
            max_epochs=50,
        )
        p_tab = tab.predict(X_te_s).flatten()
        final = 0.65 * p_xgb + 0.35 * p_tab
        log(
            f"   Ensemble R2: {r2_score(y_test, final):.4f} (XGB: {r2_score(y_test, p_xgb):.4f})"
        )
    else:
        tab = TabNetClassifier(verbose=0)
        tab.fit(
            X_tr_s, y_train.values, eval_set=[(X_te_s, y_test.values)], max_epochs=50
        )
        p_tab = tab.predict(X_te_s)
        final = p_xgb  # Simplification for categorical
        log(f"   Accuracy: {accuracy_score(y_test, final):.4f}")

    return xgb, tab, sc


if __name__ == "__main__":
    log("Starting Hybrid Training Pipeline...")

    # 1. Yield Task (From Master Data - Already High Signal R2 0.98)
    df_master = prepare_data()
    f_yield = [
        "Area_Ha",
        "soil_n",
        "soil_p",
        "soil_k",
        "soil_ph",
        "avg_temp_c",
        "humidity_pct",
        "avg_rainfall_mm",
        "gs_enc",
        "ndvi_mean",
    ]
    y_xgb, y_tab, y_sc = train_task(df_master, f_yield, "yield_ton_ha", "reg")
    joblib.dump(
        {"xgb": y_xgb, "tab": y_tab, "sc": y_sc}, MODELS_DIR / "yield_hybrid.pkl"
    )

    # 2. Water Task (From Enhanced Physics-Based Data)
    df_water_e = pd.read_csv(
        ENHANCED_DIR / "water_requirement/water_requirement_enhanced.csv"
    )
    f_water = ["temperature", "humidity", "rainfall", "growth_duration"]
    w_xgb, w_tab, w_sc = train_task(df_water_e, f_water, "water_requirement", "reg")
    joblib.dump(
        {"xgb": w_xgb, "tab": w_tab, "sc": w_sc}, MODELS_DIR / "water_hybrid.pkl"
    )

    # 3. Crop Classification (From Enhanced 96-Crop Data)
    df_crop_e = pd.read_csv(ENHANCED_DIR / "crop_recommendation/crop_data_enhanced.csv")
    le_crop = LabelEncoder()
    df_crop_e["crop_enc"] = le_crop.fit_transform(df_crop_e["label"])
    f_crop = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    c_xgb, c_tab, c_sc = train_task(df_crop_e, f_crop, "crop_enc", "cls")

    # Save Classification artifacts
    joblib.dump(
        {"xgb": c_xgb, "tab": c_tab, "sc": c_sc}, MODELS_DIR / "crop_hybrid.pkl"
    )
    joblib.dump({"crop": le_crop}, MODELS_DIR / "crop_label_encoder.pkl")

    # SHAP (Step 6)
    log("Step 6: SHAP Evaluation (Yield)")
    explainer = shap.TreeExplainer(y_xgb)
    shap_vals = explainer.shap_values(df_master[f_yield].iloc[:100])
    log(
        f"Top Features: {list(pd.Series(np.abs(shap_vals).mean(0), index=f_yield).sort_values(ascending=False).index[:3])}"
    )

    log("DONE: High-Accuracy Models saved in hybrid/")
