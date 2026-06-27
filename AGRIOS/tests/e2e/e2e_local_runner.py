#!/usr/bin/env python3
import json
import sys
import time
from fastapi.testclient import TestClient

from agrisense_app.backend import main as backend_main

app = getattr(backend_main, "app")
client = TestClient(app)

tests = []

def run_test(name, fn):
    print(f"-> {name}")
    try:
        ok, info = fn()
        print("   result:", "PASS" if ok else "FAIL")
        if info is not None:
            print("   info:", json.dumps(info)[:1000])
        tests.append((name, ok, info))
    except Exception as e:
        print("   exception:", e)
        tests.append((name, False, {"exception": str(e)}))

def test_health():
    r = client.get("/health")
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text}
    return (r.status_code == 200), body

def test_recommend_irrigation():
    payload = {
        "field_size": 5.0,
        "soil_type": "loamy",
        "current_moisture": 42.0,
        "temperature": 28.5,
        "humidity": 65.0,
        "crop_type": "tomato"
    }
    r = client.post("/api/v1/recommendations/irrigation", json=payload)
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text}
    ok = r.status_code == 200 and isinstance(body, dict) and ("recommendations" in body or "objectives" in body)
    return ok, body

def test_recommend_fertilizer():
    payload = {
        "field_size": 5.0,
        "crop_type": "tomato",
        "current_n": 80.0,
        "current_p": 40.0,
        "current_k": 70.0
    }
    r = client.post("/api/v1/recommendations/fertilizer", json=payload)
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text}
    ok = r.status_code == 200 and isinstance(body, dict) and ("recommendations" in body or "objectives" in body)
    return ok, body

def test_predict_explain():
    payload = {
        "temperature": 32.5,
        "humidity": 45.0,
        "soil_moisture": 35.0,
        "ph_level": 6.8,
        "nitrogen": 120.0,
        "phosphorus": 60.0,
        "potassium": 150.0
    }
    r = client.post("/api/v1/predict/explain", json=payload)
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text}
    ok = r.status_code == 200 and isinstance(body, dict) and ("prediction" in body or "explanation" in body)
    return ok, body

def test_chatbot_greeting():
    r = client.get("/chatbot/greeting", params={"language": "en"})
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text}
    ok = r.status_code == 200 and isinstance(body, dict) and ("greeting" in body or "message" in body)
    return ok, body

def test_chatbot_ask():
    payload = {"question": "How much water for 1 ha of tomato?", "top_k": 5, "session_id": "test-session", "language": "en", "include_sources": True}
    r = client.post("/chatbot/ask", json=payload)
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text}
    ok = r.status_code == 200 and isinstance(body, dict) and ("results" in body or "answer" in body)
    return ok, body

def test_vlm_status():
    r = client.get("/api/vlm/status")
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text}
    ok = r.status_code == 200 and isinstance(body, dict)
    return ok, body

if __name__ == "__main__":
    run_test("Health", test_health)
    time.sleep(0.5)
    run_test("Recommend: Irrigation", test_recommend_irrigation)
    run_test("Recommend: Fertilizer", test_recommend_fertilizer)
    run_test("Predict+Explain", test_predict_explain)
    run_test("Chatbot: Greeting", test_chatbot_greeting)
    run_test("Chatbot: Ask", test_chatbot_ask)
    run_test("VLM: Status", test_vlm_status)

    passed = sum(1 for t in tests if t[1])
    total = len(tests)
    score = round((passed / total) * 100, 1)

    print("\nSummary:")
    for name, ok, info in tests:
        print(f"- {name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nE2E Score (0-100): {score}  ({passed}/{total} tests passed)")
    sys.exit(0 if passed == total else 2)
