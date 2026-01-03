#!/usr/bin/env python3
import json
import sys
import time
from urllib import request, parse, error

BASE = "http://localhost:8004"
TIMEOUT = 10

tests = []

def http_get(path, params=None):
    url = BASE + path
    if params:
        url += "?" + parse.urlencode(params)
    req = request.Request(url)
    try:
        with request.urlopen(req, timeout=TIMEOUT) as r:
            return r.getcode(), json.load(r)
    except error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {"detail": str(e)}
    except Exception as e:
        return None, {"error": str(e)}

def http_post(path, body):
    url = BASE + path
    data = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=TIMEOUT) as r:
            return r.getcode(), json.load(r)
    except error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {"detail": str(e)}
    except Exception as e:
        return None, {"error": str(e)}

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
    code, body = http_get("/health")
    return (code == 200), body

def test_recommend_irrigation():
    payload = {
        "field_size": 5.0,
        "soil_type": "loamy",
        "current_moisture": 42.0,
        "temperature": 28.5,
        "humidity": 65.0,
        "crop_type": "tomato"
    }
    code, body = http_post("/api/v1/recommendations/irrigation", payload)
    ok = code == 200 and isinstance(body, dict) and ("recommendations" in body or "objectives" in body)
    return ok, body

def test_recommend_fertilizer():
    payload = {
        "field_size": 5.0,
        "crop_type": "tomato",
        "current_n": 80.0,
        "current_p": 40.0,
        "current_k": 70.0
    }
    code, body = http_post("/api/v1/recommendations/fertilizer", payload)
    ok = code == 200 and isinstance(body, dict) and ("recommendations" in body or "objectives" in body)
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
    code, body = http_post("/api/v1/predict/explain", payload)
    ok = code == 200 and isinstance(body, dict) and ("prediction" in body or "explanation" in body)
    return ok, body

def test_chatbot_greeting():
    code, body = http_get("/chatbot/greeting", {"language": "en"})
    ok = code == 200 and isinstance(body, dict) and ("greeting" in body or "message" in body)
    return ok, body

def test_chatbot_ask():
    payload = {"question": "How much water for 1 ha of tomato?", "top_k": 5, "session_id": "test-session", "language": "en", "include_sources": True}
    code, body = http_post("/chatbot/ask", payload)
    ok = code == 200 and isinstance(body, dict) and ("results" in body or "answer" in body)
    return ok, body

def test_vlm_status():
    code, body = http_get("/api/vlm/status")
    ok = code == 200 and isinstance(body, dict)
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
    score = round((passed / total) * 10, 1)

    print("\nSummary:")
    for name, ok, info in tests:
        print(f"- {name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nE2E Score (0-10): {score}  ({passed}/{total} tests passed)")
    if score < 8.0:
        print("Recommendations: check missing ML dependencies, provide model artifacts in ml_models/, and ensure image uploads for VLM tests.")
    sys.exit(0 if passed == total else 2)
