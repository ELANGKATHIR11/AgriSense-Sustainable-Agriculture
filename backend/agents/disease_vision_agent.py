# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

# backend/agents/disease_vision_agent.py
import json
from google.antigravity import Agent, LocalAgentConfig

# Import custom tools (will be created later)
from backend.tools.yolo11_tool import detect_image
from backend.tools.vrag_tool import retrieve_visual_context
from backend.tools.agridb_tool import query_agridb

class DiseaseVisionAgent(Agent):
    """Agent that composes YOLO11 VLM detection and VRAG retrieval.

    Uses the offline Ollama model `qwen2.5:1.5b-instruct`.
    """
    def __init__(self):
        config = LocalAgentConfig(
            model_name="qwen2.5:1.5b-instruct",
            offline=True,
            allow_cloud=False,
        )
        super().__init__(config=config)
        # Register custom tools
        self.register_tool(detect_image)
        self.register_tool(retrieve_visual_context)
        self.register_tool(query_agridb)

    async def run_detection(self, payload: dict):
        """Run the full pipeline on an image payload.

        Returns a dict with detections, visual context and any advisory data.
        """
        detections = await self.run_tool("detect_image", payload)
        context = await self.run_tool("retrieve_visual_context", payload)
        advisory = await self.run_tool("query_agridb", {"query": "latest_advisory"})
        return {"detections": detections, "context": context, "advisory": advisory}
