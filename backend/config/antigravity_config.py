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

# Define the local Ollama model endpoint
OLLAMA_ENDPOINT = ModelEndpoint(
    url="http://localhost:11434/v1",
    # Ollama's v1 API follows OpenAI style; Antigravity can use it as a generic endpoint
    # The model name matches the installed Ollama model
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
