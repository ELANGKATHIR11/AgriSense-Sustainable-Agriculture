"""
ESP32 Edge Intelligence and Security Configuration
TLS/MQTT security, autonomous operation, offline buffering
"""

# ===== ESP32 SECURITY CONFIGURATION =====

# Device Authentication
DEVICE_CERT_PATH = "/certs/device_cert.pem"
DEVICE_KEY_PATH = "/certs/device_key.pem"
CA_CERT_PATH = "/certs/ca_cert.pem"

# MQTT Configuration
MQTT_BROKER = "mqtt.agrisense.ai"
MQTT_PORT = 8883  # TLS port
MQTT_QOS = 1  # At least once delivery
MQTT_KEEPALIVE = 60

# TLS Configuration
MQTT_TLS_ENABLED = True
MQTT_TLS_VERSION = "TLSv1.2"
MQTT_REQUIRE_CERTIFICATE = True

# ===== EDGE INTELLIGENCE =====

# Autonomous Operation
EDGE_AUTONOMOUS_MODE = True  # Operate independently when cloud unavailable
EDGE_THRESHOLD_DETECTION = True  # Local threshold alerts

# Sensor Thresholds (for local detection)
TEMP_CRITICAL_HIGH = 45.0  # °C
TEMP_CRITICAL_LOW = 5.0    # °C
MOISTURE_CRITICAL_LOW = 15.0  # %
MOISTURE_WARNING_LOW = 30.0   # %

# Irrigation Control (local mode)
AUTO_VALVE_CONTROL = False  # Set True for autonomous irrigation
VALVE_ACTIVATION_MOISTURE = 25.0  # % - activate valve below this
VALVE_DURATION_SECONDS = 1800  # 30 minutes default

# ===== OFFLINE BUFFERING =====

OFFLINE_BUFFER_SIZE = 1000  # Max readings to buffer
OFFLINE_BUFFER_PATH = "/data/buffer.json"
RETRY_INTERVAL_SECONDS = 60  # Retry cloud connection every 60s

# ===== WATCHDOG CONFIGURATION =====

WATCHDOG_ENABLED = True
WATCHDOG_TIMEOUT_SECONDS = 30  # Reset if hung for 30s

# ===== FIRMWARE SECURITY =====

# OTA Update Configuration
OTA_ENABLED = True
OTA_SERVER = "https://ota.agrisense.ai"
OTA_VERIFY_SIGNATURE = True  # Verify firmware signature before update
OTA_AUTO_UPDATE = False  # Manual approval required

# Secure Boot
SECURE_BOOT_ENABLED = False  # Enable in production hardware
FLASH_ENCRYPTION_ENABLED = False  # Enable in production hardware

# ===== DEVICE PROVISIONING =====

# Factory Provisioning
DEVICE_ID_PREFIX = "AGRI"  # e.g., AGRI_001, AGRI_002
PROVISION_PIN_REQUIRED = True  # Require PIN for first-time setup

# ===== POWER MANAGEMENT =====

DEEP_SLEEP_ENABLED = True
SLEEP_INTERVAL_SECONDS = 300  # Wake every 5 minutes
LOW_POWER_MODE_BATTERY_THRESHOLD = 20  # % - enter low power below this

# ===== SENSOR CONFIGURATION =====

SENSOR_READING_INTERVAL = 60  # seconds
SENSOR_CALIBRATION_ENABLED = True
SENSOR_OUTLIER_REJECTION = True
SENSOR_SMOOTHING_WINDOW = 5  # readings

# ===== COMMUNICATION =====

# WiFi Configuration
WIFI_SSID = "AgriSense_Network"
WIFI_PASSWORD = ""  # Set via provisioning
WIFI_RECONNECT_ATTEMPTS = 5
WIFI_RECONNECT_DELAY_SECONDS = 10

# Fallback Communication
LORA_ENABLED = False  # Enable LoRa mesh for disaster scenarios
LORA_FREQUENCY = 915.0  # MHz (US frequency)
LORA_SPREADING_FACTOR = 7
LORA_BANDWIDTH = 125000  # Hz
