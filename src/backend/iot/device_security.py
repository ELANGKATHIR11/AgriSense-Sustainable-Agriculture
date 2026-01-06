"""
IoT Device Authentication & Security
Secures ESP32, Arduino, and MQTT communication
"""
import logging
import hashlib
import hmac
import secrets
from typing import Tuple, Optional, Dict
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class DeviceAuthentication:
    """
    Handle ESP32/Arduino device authentication.
    Uses HMAC-SHA256 for request signing.
    """

    def __init__(self, secret_key: str, max_age_seconds: int = 300):
        self.secret_key = secret_key
        self.max_age_seconds = max_age_seconds
        self.device_keys: Dict[str, str] = {}  # device_id -> secret_key

    def register_device(self, device_id: str) -> Tuple[bool, str]:
        """
        Register new IoT device and generate secret key.
        """
        if device_id in self.device_keys:
            logger.warning(f"Device {device_id} already registered")
            return False, "Device already registered"

        # Generate cryptographically secure secret key
        device_secret = secrets.token_urlsafe(32)
        self.device_keys[device_id] = device_secret

        logger.info(f"Device {device_id} registered successfully")
        return True, device_secret

    def verify_request(
        self, device_id: str, signature: str, data: str, timestamp: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify device request using HMAC signature and timestamp.
        Returns: (is_valid, error_message)
        """
        # Check device exists
        if device_id not in self.device_keys:
            return False, f"Unknown device: {device_id}"

        # Verify timestamp (prevent replay attacks)
        try:
            req_timestamp = datetime.fromisoformat(timestamp)
            now = datetime.utcnow()

            if now - req_timestamp > timedelta(seconds=self.max_age_seconds):
                return False, "Request timestamp too old (replay attack)"

        except ValueError as e:
            return False, f"Invalid timestamp format: {e}"

        # Verify HMAC signature
        device_secret = self.device_keys[device_id]
        message = f"{data}:{timestamp}"

        expected_signature = hmac.new(
            device_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            logger.warning(f"Invalid signature from device {device_id}")
            return False, "Invalid signature"

        return True, None

    def rotate_device_key(self, device_id: str) -> Tuple[bool, Optional[str]]:
        """
        Rotate device secret key (refresh).
        """
        if device_id not in self.device_keys:
            return False, f"Unknown device: {device_id}"

        new_secret = secrets.token_urlsafe(32)
        old_secret = self.device_keys[device_id]
        self.device_keys[device_id] = new_secret

        logger.info(f"Key rotated for device {device_id}")
        return True, new_secret


class MQTTSecurityConfig:
    """
    MQTT TLS/SSL security configuration for IoT sensors.
    """

    def __init__(
        self,
        broker_host: str,
        broker_port: int = 8883,
        ca_cert_path: Optional[str] = None,
        client_cert_path: Optional[str] = None,
        client_key_path: Optional[str] = None,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.ca_cert_path = ca_cert_path
        self.client_cert_path = client_cert_path
        self.client_key_path = client_key_path
        self.use_tls = True
        self.tls_version = "tlsv1_2"
        self.insecure_skip_verify = False

    def get_mqtt_config(self) -> Dict[str, any]:
        """
        Get MQTT client configuration dictionary.
        """
        config = {
            "broker_host": self.broker_host,
            "broker_port": self.broker_port,
            "use_tls": self.use_tls,
            "tls_version": self.tls_version,
        }

        if self.ca_cert_path:
            config["ca_certs"] = self.ca_cert_path

        if self.client_cert_path and self.client_key_path:
            config["certfile"] = self.client_cert_path
            config["keyfile"] = self.client_key_path

        return config

    def validate_certificate(
        self, cert_path: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate certificate exists and is readable.
        """
        try:
            with open(cert_path, "r") as f:
                cert_content = f.read()
                if "BEGIN CERTIFICATE" not in cert_content:
                    return False, "Invalid certificate format"
            return True, None
        except FileNotFoundError:
            return False, f"Certificate file not found: {cert_path}"
        except Exception as e:
            return False, str(e)


class SensorDataValidator:
    """
    Validate IoT sensor data at ingestion point.
    Prevents injection of invalid/malicious sensor readings.
    """

    # Valid ranges for common sensors
    SENSOR_RANGES = {
        "temperature": (-50, 150),  # Celsius
        "humidity": (0, 100),  # Percentage
        "soil_moisture": (0, 100),  # Percentage
        "soil_ph": (0, 14),  # pH scale
        "soil_nitrogen": (0, 1000),  # ppm
        "soil_phosphorus": (0, 1000),  # ppm
        "soil_potassium": (0, 1000),  # ppm
        "light_intensity": (0, 100000),  # lux
        "co2_level": (0, 5000),  # ppm
        "air_pressure": (800, 1200),  # hPa
    }

    SENSOR_UNITS = {
        "temperature": "°C",
        "humidity": "%",
        "soil_moisture": "%",
        "soil_ph": "pH",
        "soil_nitrogen": "ppm",
        "soil_phosphorus": "ppm",
        "soil_potassium": "ppm",
        "light_intensity": "lux",
        "co2_level": "ppm",
        "air_pressure": "hPa",
    }

    @classmethod
    def validate_sensor_reading(
        cls, sensor_type: str, value: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate single sensor reading.
        Returns: (is_valid, error_message)
        """
        if sensor_type not in cls.SENSOR_RANGES:
            return False, f"Unknown sensor type: {sensor_type}"

        min_val, max_val = cls.SENSOR_RANGES[sensor_type]

        # Check type
        if not isinstance(value, (int, float)):
            return False, f"Sensor value must be numeric, got {type(value)}"

        # Check for NaN/Inf
        if isinstance(value, float):
            if not (-float("inf") < value < float("inf")):
                return False, f"Invalid sensor value (NaN/Inf): {value}"

        # Check range
        if not (min_val <= value <= max_val):
            return (
                False,
                f"{sensor_type}={value} out of range [{min_val}, {max_val}] {cls.SENSOR_UNITS[sensor_type]}",
            )

        return True, None

    @classmethod
    def validate_batch_readings(
        cls, readings: Dict[str, float]
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, float]]]:
        """
        Validate batch of sensor readings.
        Returns: (is_valid, error_message, validated_readings)
        """
        validated = {}

        for sensor_type, value in readings.items():
            is_valid, error = cls.validate_sensor_reading(sensor_type, value)

            if not is_valid:
                return False, error, None

            validated[sensor_type] = value

        return True, None, validated

    @classmethod
    def sanitize_reading(
        cls, sensor_type: str, value: float, clip: bool = True
    ) -> float:
        """
        Sanitize sensor reading by clipping to valid range.
        """
        if sensor_type not in cls.SENSOR_RANGES:
            return value

        min_val, max_val = cls.SENSOR_RANGES[sensor_type]

        if clip:
            return max(min_val, min(value, max_val))

        return value


class FirmwareIntegrity:
    """
    Verify firmware integrity and prevent unauthorized updates.
    """

    def __init__(self):
        self.firmware_signatures: Dict[str, str] = {}  # version -> sha256

    def register_firmware(self, version: str, file_path: str) -> Tuple[bool, str]:
        """
        Register firmware version with SHA256 signature.
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            signature = sha256_hash.hexdigest()
            self.firmware_signatures[version] = signature

            logger.info(f"Firmware {version} registered with signature {signature[:16]}...")
            return True, signature

        except Exception as e:
            logger.error(f"Failed to register firmware: {e}")
            return False, str(e)

    def verify_firmware(self, version: str, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Verify firmware integrity against registered signature.
        """
        if version not in self.firmware_signatures:
            return False, f"Unknown firmware version: {version}"

        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            calculated_sig = sha256_hash.hexdigest()
            expected_sig = self.firmware_signatures[version]

            if not hmac.compare_digest(calculated_sig, expected_sig):
                logger.error(f"Firmware {version} signature mismatch")
                return False, "Firmware signature verification failed"

            logger.info(f"Firmware {version} verified successfully")
            return True, None

        except Exception as e:
            logger.error(f"Firmware verification error: {e}")
            return False, str(e)

    def update_firmware_allowed(
        self, device_id: str, current_version: str, target_version: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if firmware update is allowed (prevent downgrades by default).
        """
        if target_version not in self.firmware_signatures:
            return False, f"Target firmware version not registered: {target_version}"

        # Prevent downgrade attacks
        try:
            current_ver = tuple(map(int, current_version.split(".")))
            target_ver = tuple(map(int, target_version.split(".")))

            if target_ver < current_ver:
                logger.warning(
                    f"Downgrade attempt on device {device_id}: {current_version} -> {target_version}"
                )
                return False, "Firmware downgrade not allowed"

        except ValueError:
            logger.warning(f"Invalid version format for comparison")

        return True, None
