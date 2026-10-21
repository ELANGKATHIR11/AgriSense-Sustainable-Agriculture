import pytest
import io
import base64
from fastapi.testclient import TestClient
from PIL import Image
import sys
import os

# Add python backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.main import app

client = TestClient(app)

def test_root_and_health():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "device" in data

    res2 = client.get("/health")
    assert res2.status_code == 200
    assert res2.json()["status"] == "healthy"

def test_get_sensors():
    res = client.get("/api/sensors")
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "devices" in data
    assert "history24h" in data
    assert "irrigation" in data
    assert len(data["devices"]) >= 4
    assert len(data["history24h"]) == 24

def test_get_status():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["totalDevices"] >= 4
    assert data["aiConfigured"] is True

def test_device_config():
    payload = {
        "samplingIntervalSec": 45,
        "moistureThresholdPct": 42.0,
        "calibrationOffsetPct": 0.2
    }
    res = client.post("/api/devices/ESP32-NODE-101/config", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["device"]["samplingIntervalSec"] == 45
    assert data["device"]["moistureThresholdPct"] == 42.0

def test_irrigation_control():
    payload = {
        "zoneId": "ZONE-A",
        "action": "start",
        "durationMin": 20
    }
    res = client.post("/api/irrigation/control", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["confirmed"] is True
    assert data["zone"]["valveStatus"] == "active"
    assert data["irrigation"]["pumpStatus"] == "active"

def test_crop_recommendation():
    payload = {
        "nitrogen": 130.0,
        "phosphorus": 65.0,
        "potassium": 55.0,
        "temperature": 26.0,
        "humidity": 78.0,
        "ph": 6.5,
        "rainfall": 1150.0
    }
    res = client.post("/api/recommendations/crop", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 3
    assert "modelSource" in data
    top_rec = data["recommendations"][0]
    assert "cropName" in top_rec
    assert top_rec["confidencePct"] > 0

def test_fertilizer_recommendation():
    payload = {
        "cropType": "Paddy (CR 1009)",
        "nitrogen": 80.0,
        "phosphorus": 35.0,
        "potassium": 40.0,
        "soilPh": 6.5,
        "soilMoisture": 52.0
    }
    res = client.post("/api/recommendations/fertilizer", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["cropType"] == "Paddy (CR 1009)"
    assert "deficitsKgHa" in data
    assert data["deficitsKgHa"]["nitrogen"] > 0
    assert len(data["schedule"]) > 0

def test_disease_analyze():
    # Create test synthetic leaf image
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "imageBase64": b64_str,
        "mimeType": "image/jpeg",
        "cropHint": "Tomato",
        "language": "en"
    }
    res = client.post("/api/disease/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "diseaseName" in data
    assert "confidencePct" in data
    assert data["isModelPrediction"] is True
    assert "organicTreatment" in data
    assert "chemicalTreatment" in data

def test_digital_twin():
    res = client.get("/api/analytics/digital-twin?zoneId=ZONE-A&days=7")
    assert res.status_code == 200
    data = res.json()
    assert data["zoneId"] == "ZONE-A"
    assert "waterTwin" in data
    assert "soilTwin" in data
    assert "cropTwin" in data
    assert len(data["series"]) == 7

def test_assistant_chat():
    payload = {
        "message": "how to control aphid infestation in mustard?",
        "language": "en"
    }
    res = client.post("/api/assistant/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert len(data["reply"]) > 10
    assert len(data["sources"]) > 0

def test_operations_overview_and_actions():
    res = client.get("/api/operations/overview")
    assert res.status_code == 200
    data = res.json()
    assert "systemHealth" in data
    assert "mlOps" in data
    assert "dataOps" in data
    assert "llmOps" in data
    assert "dependencyGraph" in data

    # Settings update
    res_s = client.post("/api/operations/settings", json={"timezone": "Asia/Kolkata", "maxContinuousIrrigationMin": 40})
    assert res_s.status_code == 200
    assert res_s.json()["success"] is True

    # Hardware command
    res_h = client.post("/api/operations/hardware-command", json={"deviceId": "ESP32-NODE-101", "command": "wake_ping"})
    assert res_h.status_code == 200
    assert res_h.json()["success"] is True

    # DataOps validate
    res_v = client.post("/api/operations/dataops/validate")
    assert res_v.status_code == 200
    assert res_v.json()["success"] is True
