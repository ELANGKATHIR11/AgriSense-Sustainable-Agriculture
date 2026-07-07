# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

import json
import os
from datetime import datetime

class VisionModelRegistry:
    def __init__(self, registry_file: str = "ml/models/vision_registry.json"):
        self.registry_file = registry_file
        self.models = {}
        self.load_registry()

    def load_registry(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    self.models = json.load(f)
            except Exception:
                self.models = {}
        else:
            self.models = {
                "active_model_id": "smolvlm-v1",
                "registry": {
                    "smolvlm-v1": {
                        "name": "PlantDisease-SmolVLM-3B",
                        "version": "v1.0.0",
                        "accuracy": 0.923,
                        "f1_score": 0.919,
                        "status": "active",
                        "last_retrained": datetime.utcnow().isoformat()
                    }
                }
            }
            self.save_registry()

    def save_registry(self):
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        with open(self.registry_file, "w") as f:
            json.dump(self.models, f, indent=4)

    def register_model(self, model_id: str, name: str, version: str, accuracy: float, f1_score: float):
        self.models["registry"][model_id] = {
            "name": name,
            "version": version,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "status": "staging",
            "last_retrained": datetime.utcnow().isoformat()
        }
        self.save_registry()

    def promote_model(self, model_id: str):
        if model_id in self.models["registry"]:
            # Retire current active
            current_active = self.models.get("active_model_id")
            if current_active and current_active in self.models["registry"]:
                self.models["registry"][current_active]["status"] = "retired"
            
            self.models["registry"][model_id]["status"] = "active"
            self.models["active_model_id"] = model_id
            self.save_registry()
            print(f"Model {model_id} promoted to ACTIVE.")
        else:
            print("Model ID not found in registry.")

    def rollback_model(self, previous_model_id: str):
        self.promote_model(previous_model_id)
