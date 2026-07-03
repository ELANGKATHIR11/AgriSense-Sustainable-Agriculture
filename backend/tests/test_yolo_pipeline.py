from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_yolo_detect_mock_endpoint():
    """Verify YOLO detect endpoint processes images and returns structured box coordinates."""
    # Staged image: small green pixel base64 string
    sample_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    response = client.post("/api/vision/yolo/detect", json={"imageBase64": sample_b64})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "detections" in data
    assert len(data["detections"]) > 0
    assert "class_name" in data["detections"][0]
    assert "box" in data["detections"][0]


def test_yolo_crop_regions_endpoint():
    """Verify YOLO crop regions endpoint splits detected bounding boxes into cropped ROIs."""
    sample_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    response = client.post("/api/vision/yolo/regions", json={"imageBase64": sample_b64})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "regions" in data
    assert len(data["regions"]) > 0
    assert "imageBase64" in data["regions"][0]
