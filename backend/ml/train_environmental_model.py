import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_model(dataset_path, model_output_path):
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Check expected columns
    expected_cols = [
        "temperature",
        "humidity",
        "rainfall",
        "soil_pH",
        "disease_present",
    ]
    if not all(col in df.columns for col in expected_cols):
        print(
            f"Error: Dataset missing required columns. Expected: {expected_cols}"
        )
        print(f"Found: {df.columns.tolist()}")
        return

    X = df[["temperature", "humidity", "rainfall", "soil_pH"]]
    y = df["disease_present"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Environmental Disease Model"
    )
    # Default paths assume script is running from backend/ml or similar context, but we use robust paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))

    default_dataset = os.path.join(
        project_root, "diseases", "plant_disease_dataset.csv"
    )
    default_model_out = os.path.join(
        base_dir, "models", "environmental_model.joblib"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_model_out,
        help="Path to save trained model",
    )

    args = parser.parse_args()

    train_model(args.dataset, args.output)
