# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

"""
AGRISENSE MLOps Engine - MLflow Experiment Tracker & Model Registry
Dynamically integrates with MLflow local runs and falls back to a persistent JSON-based registry.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Any

RUNS_DB_PATH = "ml/models/mlflow_runs.json"
os.makedirs(os.path.dirname(RUNS_DB_PATH), exist_ok=True)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowTracker:
    def __init__(self, experiment_name: str = "AgriSense_Modernization"):
        self.experiment_name = experiment_name
        self.local_runs = self._load_local_runs()
        self.active_run_id = None

        if MLFLOW_AVAILABLE:
            try:
                # Set tracking URI to local file system store
                tracking_dir = os.path.abspath("ml/models/mlruns")
                os.makedirs(tracking_dir, exist_ok=True)
                mlflow.set_tracking_uri(f"file:///{tracking_dir.replace('\\', '/')}")
                mlflow.set_experiment(experiment_name)
            except Exception as e:
                print(f"MLflow setup failed: {e}. Falling back to custom JSON logger.")
                self._enable_fallback()
        else:
            self._enable_fallback()

    def _enable_fallback(self):
        global MLFLOW_AVAILABLE
        MLFLOW_AVAILABLE = False

    def _load_local_runs(self) -> dict:
        if os.path.exists(RUNS_DB_PATH):
            try:
                with open(RUNS_DB_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"runs": {}, "registry": {}}

    def _save_local_runs(self):
        with open(RUNS_DB_PATH, "w") as f:
            json.dump(self.local_runs, f, indent=4)

    def start_run(self, run_name: str) -> str:
        """Starts a training run and returns the run ID."""
        run_id = f"run_{int(time.time())}"
        self.active_run_id = run_id

        self.local_runs["runs"][run_id] = {
            "run_name": run_name,
            "status": "running",
            "start_time": datetime.now(timezone.utc).isoformat() + "Z",
            "end_time": None,
            "params": {},
            "metrics": {},
            "tags": {"experiment": self.experiment_name},
        }
        self._save_local_runs()

        if MLFLOW_AVAILABLE:
            try:
                mlflow.start_run(run_name=run_name)
            except Exception:
                pass
        return run_id

    def log_param(self, key: str, value: Any):
        if self.active_run_id and self.active_run_id in self.local_runs["runs"]:
            self.local_runs["runs"][self.active_run_id]["params"][key] = str(value)
            self._save_local_runs()

        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_param(key, value)
            except Exception:
                pass

    def log_metric(self, key: str, value: float):
        if self.active_run_id and self.active_run_id in self.local_runs["runs"]:
            self.local_runs["runs"][self.active_run_id]["metrics"][key] = float(value)
            self._save_local_runs()

        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metric(key, value)
            except Exception:
                pass

    def end_run(self, status: str = "FINISHED"):
        if self.active_run_id and self.active_run_id in self.local_runs["runs"]:
            run = self.local_runs["runs"][self.active_run_id]
            run["status"] = status
            run["end_time"] = datetime.now(timezone.utc).isoformat() + "Z"
            self._save_local_runs()
            self.active_run_id = None

        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except Exception:
                pass

    def register_model(self, model_name: str, run_id: str, accuracy: float, path: str):
        """Registers a trained model in the MLOps registry."""
        version = f"v{len(self.local_runs['registry'].get(model_name, {}).get('versions', [])) + 1}.0.0"

        if model_name not in self.local_runs["registry"]:
            self.local_runs["registry"][model_name] = {
                "active_version": version,
                "versions": [],
            }

        entry = {
            "version": version,
            "run_id": run_id,
            "accuracy": accuracy,
            "path": path,
            "status": "staging",
            "registered_at": datetime.now(timezone.utc).isoformat() + "Z",
        }

        self.local_runs["registry"][model_name]["versions"].append(entry)
        self.local_runs["registry"][model_name]["active_version"] = version
        # Set status of the version to active, others retired
        for v in self.local_runs["registry"][model_name]["versions"]:
            v["status"] = "active" if v["version"] == version else "retired"

        self._save_local_runs()
        print(f"Registered model {model_name} version {version}.")

        if MLFLOW_AVAILABLE:
            try:
                # Local MLflow model registration
                mlflow.register_model(f"runs:/{run_id}/model", model_name)
            except Exception:
                pass

    def promote_model(self, model_name: str, version: str):
        """Promotes a specific model version to ACTIVE status."""
        if model_name in self.local_runs["registry"]:
            reg = self.local_runs["registry"][model_name]
            found = False
            for v in reg["versions"]:
                if v["version"] == version:
                    v["status"] = "active"
                    reg["active_version"] = version
                    found = True
                else:
                    v["status"] = "retired"
            if found:
                self._save_local_runs()
                print(f"Model {model_name} promoted to {version}.")
            else:
                print(f"Version {version} not found for model {model_name}.")

    def rollback_model(self, model_name: str) -> str:
        """Rolls back the active model to the previously active/stable version."""
        if model_name in self.local_runs["registry"]:
            reg = self.local_runs["registry"][model_name]
            versions = reg["versions"]
            if len(versions) >= 2:
                # Find current active version index
                current_active = reg["active_version"]
                # Find previously active (which was retired)
                # Sort versions by registered_at
                sorted_versions = sorted(versions, key=lambda x: x["registered_at"])
                idx = next(
                    (
                        i
                        for i, v in enumerate(sorted_versions)
                        if v["version"] == current_active
                    ),
                    -1,
                )
                if idx > 0:
                    rollback_ver = sorted_versions[idx - 1]["version"]
                    self.promote_model(model_name, rollback_ver)
                    return rollback_ver
            print(f"No rollback target available for {model_name}.")
        return ""
