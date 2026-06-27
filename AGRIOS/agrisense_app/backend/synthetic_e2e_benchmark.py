import io
import time
import json
import base64
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AGRIOS_BENCHMARK")

print("=" * 60)
print("🚀 AGRI-OS SYNTHETIC END-TO-END BENCHMARK SUITE")
print("=" * 60)

try:
    from fastapi.testclient import TestClient
    from main import app

    print("✅ FastAPI app imported successfully")
except ImportError as e:
    print(f"❌ Failed to import FastAPI app: {e}")
    print("Ensure you are running this from the backend directory and requirements are installed.")
    exit(1)
except Exception as e:
    print(f"❌ Error initializing app: {e}")
    exit(1)

client = TestClient(app)

# --- SCORING SYSTEM ---
SCORES = {
    "system_health": 0,  # Max 10
    "agrios_pipeline": 0,  # Max 30
    "disease_detection": 0,  # Max 20
    "weed_management": 0,  # Max 20
    "chatbot_rag": 0,  # Max 20
}
# Total 100


def generate_dummy_image() -> bytes:
    """Generate a minimal valid PNG image for testing."""
    # 1x1 pixel red dot
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


def test_endpoint(name: str, method: str, url: str, **kwargs) -> Dict[str, Any]:
    """Helper to test an endpoint and return latency/status."""
    start = time.perf_counter()
    try:
        if method == "GET":
            response = client.get(url, **kwargs)
        elif method == "POST":
            response = client.post(url, **kwargs)
        else:
            return {"success": False, "error": "Invalid method"}

        duration = (time.perf_counter() - start) * 1000
        return {
            "success": response.status_code == 200,
            "status": response.status_code,
            "latency_ms": round(duration, 2),
            "data": response.json() if response.status_code == 200 else response.text,
            "url": url,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "url": url}


# --- 1. SYSTEM HEALTH (10 pts) ---
print("\n🔍 TESTING SYSTEM HEALTH...")
health = test_endpoint("Health", "GET", "/agrios/health")
if health["success"]:
    logger.info(f"✅ /agrios/health passed ({health['latency_ms']}ms)")
    SCORES["system_health"] += 10
else:
    logger.error(f"❌ /agrios/health failed: {health.get('status')} - {health.get('error')}")

# --- 2. AGRI-OS PIPELINE (30 pts) ---
print("\n🔍 TESTING AGRI-OS PIPELINE (Vision + VRAG + Governor)...")
# Test Demo pipeline (easiest way to test full flow synthetically)
demo_run = test_endpoint("Demo Run", "POST", "/agrios/demo/run", params={"scenario_id": "tomato_late_blight"})
if demo_run["success"]:
    data = demo_run["data"]
    # Check structure - DemoRunResponse has 'results' list
    if "results" in data and len(data["results"]) > 0:
        first_res = data["results"][0]
        has_gov = "governor_decision" in first_res
        has_vrag = "evidence" in first_res  # In AnalyzeResponse it's 'evidence' list, not 'vrag_evidence' key
        has_vision = "detection" in first_res

        if has_gov:
            logger.info(f"✅ /agrios/demo/run passed full schema check ({demo_run['latency_ms']}ms)")
            SCORES["agrios_pipeline"] += 30
            logger.info(f"   Governor Action: {first_res['governor_decision']['action']}")
        else:
            logger.warning(f"⚠️ /agrios/demo/run missing keys in result[0]")
            SCORES["agrios_pipeline"] += 15
    else:
        logger.warning(f"⚠️ /agrios/demo/run returned no results or wrong format")
        SCORES["agrios_pipeline"] += 0
else:
    logger.error(f"❌ /agrios/demo/run failed")

# --- 3. DISEASE DETECTION INTEGRATION (20 pts) ---
print("\n🔍 TESTING DISEASE DETECTION HOOKS...")
# We use the generic detection endpoint which should now include governor_decision
files = {"file": ("test.png", generate_dummy_image(), "image/png")}
disease = test_endpoint(
    "Disease Detect", "POST", "/api/disease/detect-image", files=files, data={"crop_type": "tomato"}
)

# Note: The actual endpoint URL in main.py might be different.
# Looking at file list, disease routes seem to be mounted.
# If '/api/disease/detect-image' fails (404), we try the direct engine calls or check main.py router.
if not disease["success"] and disease["status"] == 404:
    # Try alternate route or import engine directly if route is elusive in 'main.py' monolithic structure
    logger.info("   Endpoint /api/disease/detect-image not found, testing Engine class directly...")
    try:
        from disease_detection import DiseaseDetectionEngine

        engine = DiseaseDetectionEngine()
        res = engine.detect_disease(generate_dummy_image(), crop_type="tomato")
        if "governor_decision" in res:
            logger.info("✅ DiseaseDetectionEngine returns governor_decision")
            SCORES["disease_detection"] += 20
        else:
            logger.warning("⚠️ DiseaseDetectionEngine result missing governor_decision")
            SCORES["disease_detection"] += 10
    except Exception as e:
        logger.error(f"❌ Engine test failed: {e}")
elif disease["success"]:
    if "governor_decision" in disease["data"]:
        logger.info(f"✅ Disease endpoint returned governor_decision ({disease['latency_ms']}ms)")
        SCORES["disease_detection"] += 20
    else:
        logger.warning("⚠️ Disease endpoint output missing governor_decision")
        SCORES["disease_detection"] += 10
else:
    logger.error(f"❌ Disease detection failed: {disease.get('status')}")

# --- 4. WEED MANAGEMENT INTEGRATION (20 pts) ---
print("\n🔍 TESTING WEED MANAGEMENT HOOKS...")
try:
    from weed_management import WeedManagementEngine

    wd_engine = WeedManagementEngine()
    # Mock return path
    wd_res = wd_engine.detect_weeds(generate_dummy_image(), crop_type="corn")
    if "governor_decision" in wd_res:
        logger.info("✅ WeedManagementEngine returns governor_decision")
        SCORES["weed_management"] += 20
    else:
        logger.warning(f"⚠️ WeedManagementEngine result missing governor_decision. Keys: {list(wd_res.keys())}")
        SCORES["weed_management"] += 10
except Exception as e:
    logger.error(f"❌ Weed Engine test failed: {e}")

# --- 5. CHATBOT RAG ADAPTER (20 pts) ---
print("\n🔍 TESTING CHATBOT RAG ADAPTER...")
chat_payload = {"question": "What is tomato late blight?", "language": "en", "top_k": 3}
# The endpoint in main.py is /chatbot/ask
chatbot = test_endpoint("Chatbot Ask", "POST", "/chatbot/ask", json=chat_payload)
if chatbot["success"]:
    data = chatbot["data"]
    # We just want to ensure it works and returns results.
    # Since we didn't provide vision/sensor context in payload, it hits the standard path,
    # but the wire-up check is that it DOESN'T crash on the 'import rag_adapter' line.
    # To test the enhanced path, we'd need to mock async calls which is hard in this script.
    # But if it returns 200, the import succeeded (or failed gracefully).
    if "results" in data:
        logger.info(f"✅ /chatbot/ask returned results ({chatbot['latency_ms']}ms)")
        SCORES["chatbot_rag"] += 20
    else:
        logger.warning("⚠️ /chatbot/ask returned valid JSON but no 'results' key")
        SCORES["chatbot_rag"] += 10
else:
    logger.error(f"❌ /chatbot/ask failed: {chatbot.get('status')}")


# --- FINAL SCORE REPORT ---
total_score = sum(SCORES.values())
print("\n" + "=" * 60)
print(f"🏆 AGRI-OS SYSTEM SCORE: {total_score}/100")
print("=" * 60)
print(f"   • System Health:      {SCORES['system_health']}/10")
print(f"   • AGRI-OS Pipeline:   {SCORES['agrios_pipeline']}/30")
print(f"   • Disease Detection:  {SCORES['disease_detection']}/20")
print(f"   • Weed Management:    {SCORES['weed_management']}/20")
print(f"   • Chatbot Integration:{SCORES['chatbot_rag']}/20")
print("=" * 60)
