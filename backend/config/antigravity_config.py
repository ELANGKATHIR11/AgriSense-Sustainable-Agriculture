# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

# backend/config/antigravity_config.py

"""Antigravity SDK configuration for AgriSense.

This configuration sets up a LocalAgentConfig that uses a local Ollama
model (qwen2.5:1.5b-instruct) and restricts workspace access to the
backend directory. It enables tools, triggers, and policy capabilities.
"""

from pathlib import Path
from google.antigravity import LocalAgentConfig, CapabilitiesConfig, ModelTarget, ModelEndpoint

# Path to the backend workspace (restrict file operations)
BACKEND_ROOT = Path(__file__).resolve().parents[1]  # backend directory

class LocalOllamaEndpoint(ModelEndpoint):
    model: str | None = None

    def validate_endpoint(self) -> None:
        pass

# Define the local Ollama model endpoint
OLLAMA_ENDPOINT = LocalOllamaEndpoint(
    base_url="http://localhost:11434/v1",
    model="qwen2.5:1.5b-instruct",
)

# Capabilities we want
CAPS = CapabilitiesConfig(
    tools=True,        # enable tool execution
    triggers=True,    # enable background triggers
    policies=True,    # enable policy hooks
    memory=True,      # allow conversation memory
    compaction=True,  # automatic history compaction
)

# Assemble the full agent configuration
AG_CONFIG = LocalAgentConfig(
    model_target=ModelTarget(endpoint=OLLAMA_ENDPOINT),
    capabilities=CAPS,
    workspaces=[str(BACKEND_ROOT)],  # sandbox to backend only
)

# Export for import elsewhere
__all__ = ["AG_CONFIG", "BACKEND_ROOT"]
