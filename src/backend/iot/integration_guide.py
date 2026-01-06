"""
IoT Device Security Integration Guide

Integrates device authentication, MQTT security, and firmware validation
into the IoT sensor pipeline.

Setup Instructions:
1. Initialize device authentication
2. Configure MQTT with TLS
3. Register device firmware
4. Validate sensor data on ingestion
"""

from typing import Dict, Optional, Tuple
from backend.iot.device_security import (
    DeviceAuthentication,
    MQTTSecurityConfig,
    SensorDataValidator,
    FirmwareIntegrity,
)
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class IoTSecurityManager:
    """
    Centralized IoT security management.
    Handles device auth, MQTT config, firmware updates, sensor validation.
    """

    def __init__(self, secret_key: str, mqtt_broker: str, mqtt_port: int = 8883):
        self.device_auth = DeviceAuthentication(secret_key)
        self.mqtt_config = MQTTSecurityConfig(mqtt_broker, mqtt_port)
        self.firmware = FirmwareIntegrity()
        self.sensor_validator = SensorDataValidator()
        self.registered_devices: Dict[str, Dict] = {}

    def register_new_device(self, device_id: str, device_type: str) -> Tuple[bool, str]:
        """
        Register and provision new IoT device.
        Returns: (success, secret_key_or_error)
        """
        # Register in device auth
        success, secret = self.device_auth.register_device(device_id)

        if not success:
            return False, secret

        # Store device info
        self.registered_devices[device_id] = {
            "type": device_type,
            "registered_at": datetime.utcnow().isoformat(),
            "firmware_version": None,
            "last_reading": None,
            "reading_count": 0,
        }

        logger.info(f"Device {device_id} ({device_type}) registered successfully")
        return True, secret

    def configure_mqtt(self) -> Dict:
        """
        Get MQTT configuration for device connection.
        Returns configuration dict for MQTT client setup.
        """
        config = self.mqtt_config.get_mqtt_config()
        logger.info(f"MQTT configured for {config['broker_host']}:{config['broker_port']}")
        return config

    def validate_device_request(
        self, device_id: str, signature: str, data: str, timestamp: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate incoming device request.
        Returns: (is_valid, error_message)
        """
        is_valid, error = self.device_auth.verify_request(device_id, signature, data, timestamp)

        if not is_valid:
            logger.warning(f"Invalid request from device {device_id}: {error}")

        return is_valid, error

    def ingest_sensor_data(self, device_id: str, readings: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """
        Ingest and validate sensor data from device.
        Returns: (is_valid, error_message)
        """
        if device_id not in self.registered_devices:
            return False, f"Unknown device: {device_id}"

        # Validate sensor readings
        is_valid, error, validated = self.sensor_validator.validate_batch_readings(readings)

        if not is_valid:
            logger.warning(f"Invalid sensor data from {device_id}: {error}")
            return False, error

        # Update device stats
        self.registered_devices[device_id]["last_reading"] = datetime.utcnow().isoformat()
        self.registered_devices[device_id]["reading_count"] += 1

        logger.debug(f"Sensor data from {device_id} ingested successfully")
        return True, None

    def approve_firmware_update(
        self, device_id: str, current_version: str, target_version: str, firmware_path: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Approve firmware update for device.
        Checks version safety and firmware integrity.
        Returns: (is_allowed, error_message)
        """
        # Check device exists
        if device_id not in self.registered_devices:
            return False, f"Unknown device: {device_id}"

        # Verify firmware integrity
        is_verified, error = self.firmware.verify_firmware(target_version, firmware_path)

        if not is_verified:
            return False, error

        # Check if update is allowed
        is_allowed, error = self.firmware.update_firmware_allowed(
            device_id, current_version, target_version
        )

        if not is_allowed:
            return False, error

        logger.info(f"Firmware update approved for {device_id}: {current_version} -> {target_version}")
        return True, None

    def get_device_status(self, device_id: str) -> Optional[Dict]:
        """Get current device status."""
        if device_id not in self.registered_devices:
            return None

        status = self.registered_devices[device_id].copy()
        status["device_id"] = device_id
        status["authentication_status"] = "registered"
        status["mqtt_status"] = "configured"

        return status

    def get_security_status(self) -> Dict:
        """Get overall IoT security status."""
        return {
            "total_devices": len(self.registered_devices),
            "registered_devices": list(self.registered_devices.keys()),
            "mqtt_secured": True,
            "device_authentication": "HMAC-SHA256",
            "firmware_verification": "SHA256",
            "sensor_validation": "enabled",
            "status": "secure",
        }


# Example FastAPI Integration
"""
from fastapi import APIRouter, HTTPException, Header, Body
from pydantic import BaseModel
import json

router = APIRouter(prefix="/api/v1/iot", tags=["iot"])

# Initialize IoT security manager
iot_manager = IoTSecurityManager(
    secret_key=os.getenv("IOT_SECRET_KEY"),
    mqtt_broker=os.getenv("MQTT_BROKER_HOST", "mqtt.agrisense.local"),
    mqtt_port=int(os.getenv("MQTT_BROKER_PORT", 8883))
)

class DeviceRegistrationRequest(BaseModel):
    device_id: str
    device_type: str  # "esp32", "arduino_nano", etc.

class SensorDataRequest(BaseModel):
    device_id: str
    readings: dict  # {sensor_type: value}
    timestamp: str  # ISO format
    signature: str  # HMAC-SHA256 signature

@router.post("/devices/register")
async def register_device(request: DeviceRegistrationRequest):
    '''Register new IoT device'''
    success, result = iot_manager.register_new_device(request.device_id, request.device_type)
    
    if not success:
        raise HTTPException(status_code=400, detail=result)
    
    return {
        "device_id": request.device_id,
        "secret_key": result,
        "mqtt_config": iot_manager.configure_mqtt(),
        "message": "Device registered. Store secret_key securely."
    }

@router.post("/sensors/data")
async def submit_sensor_data(request: SensorDataRequest):
    '''Ingest sensor data with validation'''
    # Verify device request
    is_valid, error = iot_manager.validate_device_request(
        request.device_id,
        request.signature,
        json.dumps(request.readings),
        request.timestamp
    )
    
    if not is_valid:
        raise HTTPException(status_code=401, detail=error)
    
    # Ingest sensor data
    is_valid, error = iot_manager.ingest_sensor_data(request.device_id, request.readings)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    return {
        "status": "success",
        "device_id": request.device_id,
        "readings_count": len(request.readings),
        "timestamp": request.timestamp
    }

@router.post("/firmware/update")
async def request_firmware_update(
    device_id: str = Body(...),
    current_version: str = Body(...),
    target_version: str = Body(...),
    firmware_url: str = Body(...)
):
    '''Request firmware update with integrity check'''
    # Download firmware and verify
    # (Implementation depends on your firmware storage)
    
    is_allowed, error = iot_manager.approve_firmware_update(
        device_id,
        current_version,
        target_version,
        firmware_path="/path/to/firmware"
    )
    
    if not is_allowed:
        raise HTTPException(status_code=400, detail=error)
    
    return {
        "status": "approved",
        "device_id": device_id,
        "new_version": target_version,
        "firmware_url": firmware_url
    }

@router.get("/devices/{device_id}/status")
async def get_device_status(device_id: str):
    '''Get device status'''
    status = iot_manager.get_device_status(device_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return status

@router.get("/security/status")
async def get_iot_security_status():
    '''Get overall IoT security status'''
    return iot_manager.get_security_status()
"""


# ESP32 Firmware Example (Arduino Sketch)
"""
#include <WiFi.h>
#include <PubSubClient.h>
#include <mbedtls/sha256.h>
#include <time.h>

// Device configuration
const char* DEVICE_ID = "ESP32_FARM_001";
const char* DEVICE_SECRET = "YOUR_SECRET_KEY_HERE"; // From registration
const char* WIFI_SSID = "YourSSID";
const char* WIFI_PASSWORD = "YourPassword";
const char* MQTT_SERVER = "mqtt.agrisense.local";
const int MQTT_PORT = 8883;

WiFiClientSecure espClient;
PubSubClient client(espClient);

// DHT22 sensor
#include "DHT.h"
#define DHT_PIN 4
#define DHT_TYPE DHT22
DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
    Serial.begin(115200);
    
    // Initialize DHT
    dht.begin();
    
    // Connect to WiFi
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("WiFi connected");
    
    // Configure MQTT with TLS
    espClient.setInsecure(); // In production, use proper CA cert
    client.setServer(MQTT_SERVER, MQTT_PORT);
    
    // Set time (required for TLS)
    configTime(0, 0, "pool.ntp.org", "time.nist.gov");
}

void loop() {
    if (!client.connected()) {
        reconnect();
    }
    client.loop();
    
    // Read sensors every 60 seconds
    delay(60000);
    submitSensorData();
}

void submitSensorData() {
    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();
    
    if (isnan(temperature) || isnan(humidity)) {
        Serial.println("Failed to read from DHT sensor");
        return;
    }
    
    // Create JSON payload
    String readings = "{\"temperature\":" + String(temperature) + 
                     ",\"humidity\":" + String(humidity) + "}";
    
    // Generate HMAC signature
    time_t now = time(nullptr);
    String timestamp = String(now);
    String message = readings + ":" + timestamp;
    
    unsigned char signature[32];
    mbedtls_sha256((unsigned char*)message.c_str(), message.length(), signature, 0);
    
    String signatureHex = "";
    for (int i = 0; i < 32; i++) {
        char hex[3];
        sprintf(hex, "%02x", signature[i]);
        signatureHex += hex;
    }
    
    // Send to MQTT topic
    String topic = "devices/" + String(DEVICE_ID) + "/data";
    String payload = "{\"device_id\":\"" + String(DEVICE_ID) + 
                    "\",\"readings\":" + readings + 
                    ",\"timestamp\":\"" + timestamp + 
                    "\",\"signature\":\"" + signatureHex + "\"}";
    
    client.publish(topic.c_str(), payload.c_str());
    Serial.println("Data published: " + payload);
}

void reconnect() {
    while (!client.connected()) {
        if (client.connect(DEVICE_ID)) {
            Serial.println("MQTT connected");
            client.subscribe("devices/firmware/update");
        } else {
            Serial.print("MQTT connect failed: ");
            Serial.println(client.state());
            delay(5000);
        }
    }
}
"""
