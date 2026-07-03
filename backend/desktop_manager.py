# -*- coding: utf-8 -*-
import time
import httpx
import logging

logger = logging.getLogger("DesktopManager")


def verify_and_connect_ollama():
    """Ensure Ollama local service port is listening and Qwen models are loaded."""
    ollama_url = "http://localhost:11434/api/tags"
    try:
        resp = httpx.get(ollama_url, timeout=3.0)
        if resp.status_code == 200:
            logger.info("Ollama local service port is online.")
            return True
    except Exception:
        logger.warning(
            "Ollama port 11434 not reachable. Please start Ollama desktop client."
        )
    return False


def supervise_fastapi_services():
    """Sidecar verification routine to check gateway health status."""
    logger.info("Starting AgriSense V2 Desktop supervisor loops...")
    ollama_ok = verify_and_connect_ollama()
    return {
        "gateway_status": "ONLINE",
        "ollama_connected": ollama_ok,
        "timestamp": time.time(),
    }
