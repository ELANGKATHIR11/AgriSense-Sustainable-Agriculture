import logging
import json
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger("AgriSense-AI")

# Path to the existing ML inference script
BACKEND_DIR = Path(__file__).parent.parent
ML_DIR = BACKEND_DIR / "ml"
INFERENCE_SCRIPT = ML_DIR / "unified_inference.py"


class MLBridge:
    def __init__(self):
        if not INFERENCE_SCRIPT.exists():
            logger.error(f"ML Inference script not found at {INFERENCE_SCRIPT}")
        else:
            logger.info(f"ML Bridge connected to {INFERENCE_SCRIPT}")

    def get_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls the unified_inference.py script via subprocess.
        Input: Dictionary of features (N, P, K, temperature, etc.)
        Output: Dictionary of results (yield, crop, season, etc.)
        """
        try:
            # Prepare input JSON
            input_json = json.dumps(data)

            # Run the python script
            # We use 'python' assuming it's in the path, or we could use sys.executable
            # ideally we should use the same python environment
            process = subprocess.Popen(
                ["python", str(INFERENCE_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(ML_DIR),
            )

            stdout, stderr = process.communicate(input=input_json)

            if process.returncode != 0:
                logger.error(f"ML Script Error: {stderr}")
                return {"error": "ML inference failed", "details": stderr}

            # Parse output
            try:
                # The script might print other things, so we look for the last valid JSON line or just parse stdout
                # customized for unified_inference.py which prints exactly one JSON line at the end
                result = json.loads(stdout.strip())
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse ML output: {stdout}")
                return {"error": "Invalid ML output"}

        except Exception as e:
            logger.error(f"Bridge Execution Error: {e}")
            return {"error": str(e)}

    def get_mock_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback mock prediction for testing without full ML setup.
        """
        return {
            "water_requirement": 45.2,
            "season": "Kharif",
            "crop_group": "Cereal",
            "recommended_crop": "Rice",
            "expected_yield": 4200.5,
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        Scan backend/ml/datasets directory and return dataset metadata.
        """
        datasets_dir = ML_DIR / "datasets"
        datasets = []

        if not datasets_dir.exists():
            datasets_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Created datasets directory: {datasets_dir}")
            return datasets

        for idx, file_path in enumerate(datasets_dir.iterdir()):
            if file_path.is_file():
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)

                # Estimate records for CSV
                records = 0
                if file_path.suffix.lower() == ".csv":
                    try:
                        with open(file_path, "r") as f:
                            records = sum(1 for _ in f) - 1  # Exclude header
                    except:
                        records = 0

                datasets.append(
                    {
                        "id": str(idx + 1),
                        "name": file_path.name,
                        "type": (
                            "CSV"
                            if file_path.suffix.lower() == ".csv"
                            else (
                                "Image"
                                if file_path.suffix.lower() in [".zip", ".tar"]
                                else "Other"
                            )
                        ),
                        "size": f"{size_mb:.1f} MB",
                        "records": records,
                        "uploaded_at": datetime.fromtimestamp(stat.st_mtime).strftime(
                            "%Y-%m-%d"
                        ),
                        "status": "Ready",
                    }
                )

        return datasets

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Scan backend/ml/models and backend/ai_service/models for trained models.
        """
        models = []
        model_dirs = [ML_DIR / "models", BACKEND_DIR / "ai_service" / "models"]

        for model_dir in model_dirs:
            if not model_dir.exists():
                continue

            for idx, file_path in enumerate(model_dir.iterdir()):
                if file_path.is_file() and file_path.suffix in [
                    ".json",
                    ".pth",
                    ".pkl",
                    ".joblib",
                ]:
                    stat = file_path.stat()

                    # Determine model type
                    model_type = "Unknown"
                    if "yield" in file_path.name.lower():
                        model_type = "Regression"
                    elif (
                        "crop" in file_path.name.lower()
                        or "recommend" in file_path.name.lower()
                    ):
                        model_type = "Classification"
                    elif "vlm" in file_path.name.lower():
                        model_type = "Vision-Language"

                    models.append(
                        {
                            "id": str(len(models) + 1),
                            "name": file_path.stem.replace("_", " ").title(),
                            "version": "v1.0",
                            "type": model_type,
                            "status": "Trained",
                            "accuracy": 0.0,  # Would need metadata file
                            "last_trained": datetime.fromtimestamp(
                                stat.st_mtime
                            ).strftime("%Y-%m-%d"),
                            "dataset_id": "1",
                        }
                    )

        return models

    def trigger_training(self, model_id: str) -> Dict[str, Any]:
        """
        Trigger a training script based on model ID.
        Returns status of the training process.
        """
        try:
            # Map model IDs to training scripts
            training_scripts = {
                "1": "train_yield.py",
                "2": "train_vlm_targeted.py",
            }

            script_name = training_scripts.get(model_id)
            if not script_name:
                return {"status": "error", "message": "Unknown model ID"}

            script_path = ML_DIR / script_name
            if not script_path.exists():
                return {
                    "status": "error",
                    "message": f"Training script not found: {script_name}",
                }

            # Start training as background process
            logger.info(f"Starting training: {script_name}")
            process = subprocess.Popen(
                ["python", str(script_path)],
                cwd=str(ML_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            return {
                "status": "started",
                "message": f"Training initiated for {script_name}",
                "pid": process.pid,
            }
        except Exception as e:
            logger.error(f"Training trigger error: {e}")
            return {"status": "error", "message": str(e)}
