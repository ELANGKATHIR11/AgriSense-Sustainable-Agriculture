"""
Train NPU-friendly TensorFlow models (Conv1D-based) and export to ONNX.

This replaces Dense layers with Conv1D(kernel_size=1) patterns to improve
compatibility with Intel NPU conversion passes.

Usage: .venv_py312_npu\Scripts\python.exe tools/npu/train_tf_npu_friendly.py
"""
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


def build_npu_friendly_model(input_dim, num_classes, filters=(128, 64)):
    """Build a Conv1D-based model using kernel_size=1 to emulate Dense layers.

    Avoids large fully-connected ops that the NPU compiler sometimes fails on.
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Reshape((input_dim, 1)))

    for f in filters:
        model.add(keras.layers.Conv1D(filters=f, kernel_size=1, activation="relu"))
        model.add(keras.layers.Dropout(0.15))

    # project to num_classes channels and pool to get classification vector
    # Use GlobalAveragePooling1D which is dynamic-batch-safe and avoids explicit
    # fixed pool sizes that caused AvgPool shape issues in the NPU compiler.
    model.add(keras.layers.Conv1D(filters=num_classes, kernel_size=1, activation=None))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Activation("softmax"))

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_export(X_train, y_train, X_val, y_val, name, filters=(128, 64), epochs=60):
    model = build_npu_friendly_model(X_train.shape[1], y_train.shape[1], filters=filters)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    start = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32, callbacks=callbacks, verbose=2)
    elapsed = time.time() - start

    save_dir = MODELS_DIR / name
    # Keras 3: use model.export to create a SavedModel directory
    try:
        model.export(str(save_dir))
    except Exception:
        # fallback to legacy save if export isn't available
        model.save(save_dir, include_optimizer=False)
    model.save(MODELS_DIR / f"{name}.h5", include_optimizer=False)

    # write metrics
    val_acc = history.history.get("val_accuracy", [None])[-1]
    metrics = {"train_time_s": elapsed, "val_accuracy": float(val_acc) if val_acc is not None else None}
    with open(MODELS_DIR / f"{name}_training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved NPU-friendly TF model: {save_dir} (val_acc={metrics['val_accuracy']})")

    # Try export to ONNX using tf2onnx. Prefer the in-process API, but fall back
    # to invoking the tf2onnx CLI via `python -m tf2onnx.convert` when the API
    # doesn't support the model object (common with Sequential/Keras wrappers).
    onnx_path = MODELS_DIR / "openvino_npu" / f"{name}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tf2onnx

        try:
            spec = (tf.TensorSpec((None, X_train.shape[1]), tf.float32, name="input"),)
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
            with open(onnx_path, "wb") as f:
                f.write(model_proto.SerializeToString())
            print(f"✅ ONNX exported (tf2onnx.from_keras): {onnx_path}")
        except Exception as inner_e:
            print(f"tf2onnx.from_keras failed: {inner_e}; falling back to CLI converter")
            # Fall through to CLI below
            raise RuntimeError("fallback-to-cli")
    except Exception:
        # Run CLI: `python -m tf2onnx.convert --saved-model <save_dir> --output <onnx_path> --opset 13`
        import subprocess, sys

        try:
            cmd = [sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(save_dir), "--output", str(onnx_path), "--opset", "13"]
            print("Running tf2onnx CLI:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            print(f"✅ ONNX exported (tf2onnx CLI): {onnx_path}")
        except subprocess.CalledProcessError as cli_e:
            print(f"ONNX CLI conversion failed: {cli_e}")

    # Also try OpenVINO conversion of SavedModel (may still fail compilation)
    try:
        from openvino import convert_model
        ov_model = convert_model(str(save_dir))
        out_dir = MODELS_DIR / "openvino_npu" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        import openvino as _ov
        _ov.save_model(ov_model, str(out_dir / f"{name}.xml"))
        print(f"✅ OpenVINO IR saved: {out_dir}")
    except Exception as e:
        print(f"OpenVINO conversion failed or skipped: {e}")

    return save_dir


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

    # Train NPU-friendly models
    sm_dir = train_and_export(X_train, y_train, X_val, y_val, "crop_recommendation_tf_npu_small", filters=(64,))
    md_dir = train_and_export(X_train, y_train, X_val, y_val, "crop_recommendation_tf_npu_medium", filters=(128, 64))

    print("NPU-friendly TF training + conversion complete. Models in:", MODELS_DIR)


if __name__ == "__main__":
    main()
