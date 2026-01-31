# AgriSense ESP32 config stub for local tooling
CONFIG = {
    "ssid": "YOUR_SSID",
    "password": "YOUR_PASSWORD",
    "backend_url": "http://your-backend.local/api/sensors",
    "device_id": "esp32-stub-001"
}

if __name__ == "__main__":
    print("ESP32 config:", CONFIG)
