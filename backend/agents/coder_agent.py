# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

# -*- coding: utf-8 -*-
import httpx
import logging
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/ml/coder", tags=["Developer Agent"])

logger = logging.getLogger("CoderAgent")
OLLAMA_BASE = "http://localhost:11434"


class CoderRequest(BaseModel):
    prompt: str
    context: str = ""


@router.post("/execute")
async def run_coder_agent(payload: CoderRequest):
    ollama_payload = {
        "model": "qwen2.5:1.5b-instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the AgriSense Coder Agent. Your job is to generate clean python code, "
                    "SQL queries, bug fixes, or unit tests for the AgriSense smart agriculture platform. "
                    "Output clean, modular code with minimal conversational explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Context: {payload.context}\n\nTask: {payload.prompt}",
            },
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=ollama_payload)
            if resp.status_code == 200:
                answer = resp.json().get("message", {}).get("content", "")
                return {
                    "result": answer,
                    "agent": "CoderAgent Qwen3B",
                    "status": "SUCCESS",
                }
    except Exception as e:
        logger.warning(f"Ollama Coder model not available: {e}")

    # Fallback template
    return {
        "result": "# Coder Agent Fallback Template\ndef run_check():\n    print('Calibration OK')\n",
        "agent": "CoderAgent Emulator",
        "status": "FALLBACK",
    }
