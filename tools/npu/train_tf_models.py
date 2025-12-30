"""
Train TensorFlow models for Crop Recommendation to replace scikit-learn models.

This script trains two TensorFlow MLPs (small and medium) on
`agrisense_app/backend/Crop_recommendation.csv`, saves them as SavedModel,
and attempts to convert them to OpenVINO IR for NPU inference.

Usage: python tools/npu/train_tf_models.py
"""
import os
import time
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow import keras

ROOT = Path("agrisense_app/backend")
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_preprocess(csv_path: Path):
    df = pd.read_csv(csv_path)
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].astype(str).values

    # Label encoding
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # save artifacts
    joblib.dump(scaler, MODELS_DIR / "crop_scaler.joblib")
    joblib.dump(le, MODELS_DIR / "crop_encoder.joblib")

    y_cat = keras.utils.to_categorical(y, num_classes)
    return X, y_cat, num_classes, le


def build_mlp(input_dim, num_classes, hidden=(128, 64)):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    for h in hidden:
        model.add(keras.layers.Dense(h, activation="relu"))
        model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_save(X_train, y_train, X_val, y_val, name, hidden):
    model = build_mlp(X_train.shape[1], y_train.shape[1], hidden=hidden)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    start = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=80, batch_size=32, callbacks=callbacks, verbose=2)
    elapsed = time.time() - start

    save_dir = MODELS_DIR / name
    # save as TensorFlow SavedModel via export
    model.export(str(save_dir))
    # also save an h5 for convenience
    model.save(MODELS_DIR / f"{name}.h5", include_optimizer=False)

    # metrics
    val_acc = history.history.get("val_accuracy", [None])[-1]
    metrics = {"train_time_s": elapsed, "val_accuracy": float(val_acc) if val_acc is not None else None}
    with open(MODELS_DIR / f"{name}_training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved TF model: {save_dir} (val_acc={metrics['val_accuracy']})")
    return save_dir


def try_convert_to_openvino(saved_model_dir: Path, name: str):
    print(f"Converting {name} to OpenVINO IR if possible...")
    try:
        # openvino.convert_model should accept SavedModel directories
        from openvino import convert_model
        ov_model = convert_model(str(saved_model_dir))
        out_dir = MODELS_DIR / "openvino_npu" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        import openvino as _ov
        _ov.save_model(ov_model, str(out_dir / f"{name}.xml"))
        print(f"✅ OpenVINO IR saved: {out_dir}")
    except Exception as e:
        print(f"OpenVINO conversion failed or skipped: {e}")


def main():
    csv_candidates = [ROOT / "Crop_recommendation.csv", ROOT / "data" / "Crop_recommendation.csv"]
    csv_path = None
    for c in csv_candidates:
        if c.exists():
            csv_path = c
            break
    if csv_path is None:
        print("Dataset not found; please generate Crop_recommendation.csv first")
        return

    X, y, num_classes, le = load_and_preprocess(csv_path)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))

    # Train small and medium MLPs
    sm_dir = train_and_save(X_train, y_train, X_val, y_val, "crop_recommendation_tf_small", hidden=(64,))
    md_dir = train_and_save(X_train, y_train, X_val, y_val, "crop_recommendation_tf_medium", hidden=(128, 64))

    # try export to OpenVINO
    try_convert_to_openvino(sm_dir, "crop_recommendation_tf_small")
    try_convert_to_openvino(md_dir, "crop_recommendation_tf_medium")

    print("TF training + conversion complete. Models in:", MODELS_DIR)


if __name__ == "__main__":
    main()
