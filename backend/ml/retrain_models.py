# pyright: reportMissingTypeStubs=false
import warnings
from pathlib import Path
from typing import Any, Dict, Sequence

import joblib  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import xgboost as xgb
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.metrics import (accuracy_score,  # type: ignore
                             mean_absolute_error, r2_score)
from sklearn.model_selection import KFold, StratifiedKFold  # type: ignore
from sklearn.preprocessing import (LabelEncoder,  # type: ignore
                                   OrdinalEncoder, StandardScaler)

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None
try:
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None


BASE = Path(__file__).parent
warnings.filterwarnings("ignore")
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
    slope = 0.0
    if len(vals) >= 2:
        slope = float(np.polyfit(range(len(vals)), vals, 1)[0])
    return mean, std, slope


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
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

    ndvi_cols = ["ndvi_seq_mean", "ndvi_seq_std", "ndvi_seq_slope"]
    df[ndvi_cols] = df.apply(lambda r: pd.Series(ndvi_features(r)), axis=1)

    return df


def run_xgb_classification(X, y, name: str):
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_imp)

    # We'll attempt LightGBM and CatBoost (if available) and XGBoost.
    # Choose best model by randomized CV to avoid sklearn tag issues.
    from random import Random

    rnd = Random(42)
    cv = StratifiedKFold(n_splits=3)
    y_arr = np.asarray(y)
    le = None
    if y_arr.dtype.kind in ("U", "S", "O"):
        le = LabelEncoder()
        y_enc = le.fit_transform(y_arr)
        label_fname = f"optimized_{name}_label_encoder.pkl"
        label_path = MODELS_DIR / label_fname
        joblib.dump(le, label_path)
    else:
        y_enc = y_arr

    best_global_score = -float("inf")
    best_global_model = None

    # XGBoost search
    xgb_param_dist: Dict[str, Sequence[Any]] = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    for _ in range(6):
        params_xgb: Dict[str, Any] = {
            k: rnd.choice(v) for k, v in xgb_param_dist.items()
        }
        m = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            **params_xgb,
        )
        scores = []
        for train_idx, test_idx in cv.split(X_s, y_enc):
            Xtr, Xte = X_s[train_idx], X_s[test_idx]
            ytr, yte = y_enc[train_idx], y_enc[test_idx]
            m.fit(Xtr, ytr)
            p = m.predict(Xte)  # type: ignore[attr-defined]
            scores.append(float(accuracy_score(yte, p)))
        mean_score = float(np.mean(scores))
        if mean_score > best_global_score:
            best_global_score = mean_score
            best_global_model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                **params_xgb,
            )

    # LightGBM search (if available)
    if lgb is not None:
        lgb_param_dist: Dict[str, Sequence[Any]] = {
            "n_estimators": [100, 200, 300],
            "max_depth": [-1, 3, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        for _ in range(6):
            params_lgb: Dict[str, Any] = {
                k: rnd.choice(v) for k, v in lgb_param_dist.items()
            }
            m = lgb.LGBMClassifier(**params_lgb)
            scores = []
            for train_idx, test_idx in cv.split(X_s, y_enc):
                Xtr, Xte = X_s[train_idx], X_s[test_idx]
                ytr, yte = y_enc[train_idx], y_enc[test_idx]
                m.fit(Xtr, ytr)
                p = m.predict(Xte)  # type: ignore[attr-defined]
                scores.append(float(accuracy_score(yte, p)))
            mean_score = float(np.mean(scores))
            if mean_score > best_global_score:
                best_global_score = mean_score
                best_global_model = lgb.LGBMClassifier(**params_lgb)

    # CatBoost search (if available)
    if CatBoostClassifier is not None:
        cat_param_dist: Dict[str, Sequence[Any]] = {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "rsm": [0.6, 0.8, 1.0],
        }
        for _ in range(6):
            params_cat: Dict[str, Any] = {
                k: rnd.choice(v) for k, v in cat_param_dist.items()
            }
            m = CatBoostClassifier(**params_cat, verbose=0)
            scores = []
            for train_idx, test_idx in cv.split(X_s, y_enc):
                Xtr, Xte = X_s[train_idx], X_s[test_idx]
                ytr, yte = y_enc[train_idx], y_enc[test_idx]
                m.fit(Xtr, ytr)
                p = m.predict(Xte)  # type: ignore[attr-defined]
                scores.append(float(accuracy_score(yte, p)))
            mean_score = float(np.mean(scores))
            if mean_score > best_global_score:
                best_global_score = mean_score
                best_global_model = CatBoostClassifier(**params_cat, verbose=0)

    # Final fit on full data
    if best_global_model is None:
        # fallback to simple xgboost default
        best_global_model = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric="mlogloss"
        )
    best_global_model.fit(X_s, y_enc)
    preds = best_global_model.predict(X_s)  # type: ignore[attr-defined]
    acc = accuracy_score(y_enc, preds)
    model_fname = f"optimized_{name}_model.pkl"
    scaler_fname = f"optimized_{name}_scaler.pkl"
    imp_fname = f"optimized_{name}_imputer.pkl"
    model_path = MODELS_DIR / model_fname
    scaler_path = MODELS_DIR / scaler_fname
    imp_path = MODELS_DIR / imp_fname
    joblib.dump(best_global_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(imp, imp_path)
    return best_global_model, acc


def run_xgb_regression(X, y, name: str):
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_imp)

    # Try LightGBM, CatBoost, and XGBoost regressors and pick best by CV
    from random import Random

    rnd = Random(42)
    cv = KFold(n_splits=3)
    y_arr = np.asarray(y)

    best_global_score = -float("inf")
    best_global_model = None

    xgb_param_dist: Dict[str, Sequence[Any]] = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    for _ in range(6):
        params_xgb_reg: Dict[str, Any] = {
            k: rnd.choice(v) for k, v in xgb_param_dist.items()
        }
        m = xgb.XGBRegressor(objective="reg:squarederror", **params_xgb_reg)
        scores = []
        for train_idx, test_idx in cv.split(X_s):
            Xtr, Xte = X_s[train_idx], X_s[test_idx]
            ytr, yte = y_arr[train_idx], y_arr[test_idx]
            m.fit(Xtr, ytr)
            p = m.predict(Xte)  # type: ignore[attr-defined]
            scores.append(float(r2_score(yte, p)))
        mean_score = float(np.mean(scores))
        if mean_score > best_global_score:
            best_global_score = mean_score
            best_global_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                **params_xgb_reg,
            )

    if lgb is not None:
        lgb_param_dist: Dict[str, Sequence[Any]] = {
            "n_estimators": [100, 200, 300],
            "max_depth": [-1, 3, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        for _ in range(6):
            params_lgb_reg: Dict[str, Any] = {
                k: rnd.choice(v) for k, v in lgb_param_dist.items()
            }
            m = lgb.LGBMRegressor(**params_lgb_reg)
            scores = []
            for train_idx, test_idx in cv.split(X_s):
                Xtr, Xte = X_s[train_idx], X_s[test_idx]
                ytr, yte = y_arr[train_idx], y_arr[test_idx]
                m.fit(Xtr, ytr)
                p = m.predict(Xte)  # type: ignore[attr-defined]
                scores.append(float(r2_score(yte, p)))
            mean_score = float(np.mean(scores))
            if mean_score > best_global_score:
                best_global_score = mean_score
                best_global_model = lgb.LGBMRegressor(**params_lgb_reg)

    if CatBoostRegressor is not None:
        cat_param_dist: Dict[str, Sequence[Any]] = {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "rsm": [0.6, 0.8, 1.0],
        }
        for _ in range(6):
            params_cat_reg: Dict[str, Any] = {
                k: rnd.choice(v) for k, v in cat_param_dist.items()
            }
            m = CatBoostRegressor(**params_cat_reg, verbose=0)
            scores = []
            for train_idx, test_idx in cv.split(X_s):
                Xtr, Xte = X_s[train_idx], X_s[test_idx]
                ytr, yte = y_arr[train_idx], y_arr[test_idx]
                m.fit(Xtr, ytr)
                p = m.predict(Xte)  # type: ignore[attr-defined]
                scores.append(float(r2_score(yte, p)))
            mean_score = float(np.mean(scores))
            if mean_score > best_global_score:
                best_global_score = mean_score
                best_global_model = CatBoostRegressor(
                    **params_cat_reg, verbose=0
                )

    if best_global_model is None:
        best_global_model = xgb.XGBRegressor(objective="reg:squarederror")
    best_global_model.fit(X_s, y_arr)
    preds = best_global_model.predict(X_s)  # type: ignore[attr-defined]
    r2 = r2_score(y_arr, preds)
    mae = mean_absolute_error(y_arr, preds)
    model_path = MODELS_DIR / f"optimized_{name}_model.pkl"
    scaler_path = MODELS_DIR / f"optimized_{name}_scaler.pkl"
    imp_path = MODELS_DIR / f"optimized_{name}_imputer.pkl"
    joblib.dump(best_global_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(imp, imp_path)
    return best_global_model, r2, mae


def main():
    csv_path = (
        Path(__file__).parent.parent.parent
        / "indian_agriculture_ml_dataset.csv"
    )
    print("Loading dataset:", csv_path)
    df = load_data(csv_path)
    df = preprocess(df)

    # encode categorical columns
    cat_cols = [
        c
        for c in ["soil_texture", "growth_stage", "irrigation_method"]
        if c in df.columns
    ]
    if cat_cols:
        oe = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df[cat_cols] = oe.fit_transform(df[cat_cols].fillna("__MISSING__"))
        oe_path = MODELS_DIR / "optimized_categorical_encoder.pkl"
        joblib.dump(oe, oe_path)

    # Crop recommendation
    cr_target = "recommended_crop"
    cr_features = [
        f
        for f in [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ]
        if f in df.columns
    ]
    cr_features += [
        "ndvi_seq_mean",
        "ndvi_seq_std",
        "ndvi_seq_slope",
    ] + cat_cols
    df_cr = df.dropna(subset=cr_features + [cr_target])
    print("Optimized training Crop Recommendation rows:", len(df_cr))
    if len(df_cr) > 50:
        X_cr = df_cr[cr_features].astype(float)
        y_cr = df_cr[cr_target].astype(str)
        cr_model, cr_acc = run_xgb_classification(
            X_cr,
            y_cr,
            "crop_recommendation",
        )
        print("Crop recommendation optimized accuracy:", cr_acc)

    # Yield prediction
    y_target = "yield_kg"
    y_features = [
        f
        for f in [
            "N",
            "P",
            "K",
            "temperature",
            "rainfall",
            "water_required_m3",
        ]
        if f in df.columns
    ]
    y_features += ["ndvi_seq_mean", "ndvi_seq_slope"]
    df_y = df.dropna(subset=y_features + [y_target])
    print("Optimized training Yield Prediction rows:", len(df_y))
    if len(df_y) > 50:
        X_y = df_y[y_features].astype(float)
        y_y = df_y[y_target].astype(float)
        y_model, y_r2, y_mae = run_xgb_regression(
            X_y,
            y_y,
            "yield_prediction",
        )
        print("Yield optimized R2/MAE:", y_r2, y_mae)

    # Crop type classification
    ct_target = "crop_type"
    ct_features = [
        f
        for f in [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ]
        if f in df.columns
    ]
    ct_features += ["ndvi_seq_mean", "ndvi_seq_slope"]
    df_ct = df.dropna(subset=ct_features + [ct_target])
    print("Optimized training Crop Type rows:", len(df_ct))
    if len(df_ct) > 50:
        X_ct = df_ct[ct_features].astype(float)
        y_ct = df_ct[ct_target].astype(str)
        ct_model, ct_acc = run_xgb_classification(
            X_ct,
            y_ct,
            "crop_type",
        )
        print("Crop type optimized accuracy:", ct_acc)

    # Water requirement regression
    wr_target = "water_required_m3"
    wr_features = [
        f
        for f in ["temperature", "humidity", "rainfall", "sowing_day_of_year"]
        if f in df.columns
    ]
    wr_features += ["ndvi_seq_mean", "ndvi_seq_slope"]
    df_wr = df.dropna(subset=wr_features + [wr_target])
    print("Optimized training Water Requirement rows:", len(df_wr))
    if len(df_wr) > 50:
        X_wr = df_wr[wr_features].astype(float)
        y_wr = df_wr[wr_target].astype(float)
        wr_model, wr_r2, wr_mae = run_xgb_regression(
            X_wr,
            y_wr,
            "water_requirement",
        )
        print("Water Req optimized R2/MAE:", wr_r2, wr_mae)

    # Season classification
    st_target = "season_target" if "season_target" in df.columns else "season"
    st_features = [
        f for f in ["temperature", "rainfall", "humidity"] if f in df.columns
    ]
    st_features += ["ndvi_seq_mean", "ndvi_seq_slope"]
    df_st = df.dropna(subset=st_features + [st_target])
    print("Optimized training Season Classification rows:", len(df_st))
    if len(df_st) > 50:
        X_st = df_st[st_features].astype(float)
        y_st = df_st[st_target].astype(str)
        st_model, st_acc = run_xgb_classification(
            X_st,
            y_st,
            "season_classification",
        )
        print("Season optimized accuracy:", st_acc)

    print("Optimized training complete. Models saved to:")
    for p in sorted(MODELS_DIR.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
