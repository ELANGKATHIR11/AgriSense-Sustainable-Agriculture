"""Test ML Endpoints"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_water_optimization():
    print("=" * 50)
    print("Testing Water Optimization Endpoints")
    print("=" * 50)
    
    # Test info endpoint
    print("\n1. GET /api/ml/water-optimization/info")
    try:
        response = requests.get(f"{BASE_URL}/api/ml/water-optimization/info", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test prediction endpoint
    print("\n2. POST /api/ml/water-optimization")
    try:
        data = {
            "crop_type": "wheat",
            "soil_type": "loamy",
            "temperature": 28.5,
            "humidity": 65.0,
            "soil_moisture": 35.0,
            "rainfall_last_7_days": 12.0,
            "growth_stage": "vegetative",
            "field_area_m2": 100.0
        }
        response = requests.post(f"{BASE_URL}/api/ml/water-optimization", json=data, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")

def test_yield_prediction():
    print("\n" + "=" * 50)
    print("Testing Yield Prediction Endpoints")
    print("=" * 50)
    
    # Test info endpoint
    print("\n1. GET /api/ml/yield-prediction/info")
    try:
        response = requests.get(f"{BASE_URL}/api/ml/yield-prediction/info", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test prediction endpoint
    print("\n2. POST /api/ml/yield-prediction")
    try:
        data = {
            "crop_type": "rice",
            "soil_type": "clay",
            "nitrogen": 45.0,
            "phosphorus": 30.0,
            "potassium": 35.0,
            "temperature": 26.0,
            "humidity": 75.0,
            "rainfall": 150.0,
            "irrigation": 100.0,
            "field_area_ha": 5.0
        }
        response = requests.post(f"{BASE_URL}/api/ml/yield-prediction", json=data, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")

def test_health():
    print("\n" + "=" * 50)
    print("Testing Health Endpoint")
    print("=" * 50)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    print("\n🌾 AgriSense ML Endpoints Test Suite\n")
    test_health()
    test_water_optimization()
    test_yield_prediction()
    print("\n✅ Tests completed!")
