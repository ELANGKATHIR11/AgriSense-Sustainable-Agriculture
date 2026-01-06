"""
Sensor Data Validation with Security Checks
Prevents sensor spoofing and impossible values
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from ..config.optimization import settings

logger = logging.getLogger(__name__)


class SensorTamperingError(Exception):
    """Raised when sensor data appears to be tampered or spoofed"""
    pass


class SensorValidator:
    """
    Validates sensor readings for security and data quality.
    Implements anomaly detection and spoofing prevention.
    """
    
    def __init__(self):
        self.last_readings: Dict[str, Dict[str, Any]] = {}
        self.anomaly_counts: Dict[str, int] = {}
        
    def validate_reading(
        self,
        device_id: str,
        data: Dict[str, Any],
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Validate a sensor reading for impossible values and anomalies.
        
        Args:
            device_id: Unique device identifier
            data: Sensor reading data
            strict: If True, raise on validation errors. If False, return warnings.
            
        Returns:
            Dict with validation results and cleaned data
            
        Raises:
            SensorTamperingError: If data appears tampered
        """
        if not settings.enable_sensor_validation:
            return {"valid": True, "data": data, "warnings": []}
        
        warnings = []
        errors = []
        
        # Validate temperature
        if "temperature" in data:
            temp = data["temperature"]
            if not isinstance(temp, (int, float)):
                errors.append(f"Invalid temperature type: {type(temp)}")
            elif temp < settings.sensor_min_temp or temp > settings.sensor_max_temp:
                errors.append(
                    f"Temperature {temp}°C out of range "
                    f"[{settings.sensor_min_temp}, {settings.sensor_max_temp}]"
                )
        
        # Validate humidity
        if "humidity" in data:
            humidity = data["humidity"]
            if not isinstance(humidity, (int, float)):
                errors.append(f"Invalid humidity type: {type(humidity)}")
            elif humidity < settings.sensor_min_humidity or humidity > settings.sensor_max_humidity:
                errors.append(
                    f"Humidity {humidity}% out of range "
                    f"[{settings.sensor_min_humidity}, {settings.sensor_max_humidity}]"
                )
        
        # Validate soil moisture
        if "soil_moisture" in data or "moisture" in data:
            moisture = data.get("soil_moisture") or data.get("moisture")
            if moisture is not None:
                if not isinstance(moisture, (int, float)):
                    errors.append(f"Invalid moisture type: {type(moisture)}")
                elif moisture < settings.sensor_min_moisture or moisture > settings.sensor_max_moisture:
                    errors.append(
                        f"Soil moisture {moisture}% out of range "
                        f"[{settings.sensor_min_moisture}, {settings.sensor_max_moisture}]"
                    )
        
        # Check for impossible rapid changes (spoofing detection)
        if device_id in self.last_readings:
            last_data = self.last_readings[device_id]
            last_time = last_data.get("timestamp")
            
            if last_time and "temperature" in data and "temperature" in last_data["data"]:
                time_diff = (datetime.now() - last_time).total_seconds()
                if time_diff < 60:  # Within 1 minute
                    temp_diff = abs(data["temperature"] - last_data["data"]["temperature"])
                    # Temperature shouldn't change > 10°C per minute in natural conditions
                    if temp_diff > 10.0:
                        warnings.append(
                            f"Suspicious temperature change: {temp_diff}°C in {time_diff}s"
                        )
        
        # Store this reading for next comparison
        self.last_readings[device_id] = {
            "data": data.copy(),
            "timestamp": datetime.now()
        }
        
        # Handle errors
        if errors:
            self.anomaly_counts[device_id] = self.anomaly_counts.get(device_id, 0) + 1
            
            # Log suspicious activity
            if self.anomaly_counts[device_id] >= 3:
                logger.error(
                    f"Multiple validation failures for device {device_id}. "
                    f"Possible tampering! Errors: {errors}"
                )
            
            if strict:
                raise SensorTamperingError(
                    f"Sensor validation failed for {device_id}: {', '.join(errors)}"
                )
        
        return {
            "valid": len(errors) == 0,
            "data": data,
            "warnings": warnings,
            "errors": errors
        }
    
    def reset_device(self, device_id: str):
        """Reset validation state for a device"""
        self.last_readings.pop(device_id, None)
        self.anomaly_counts.pop(device_id, None)


# Singleton instance
sensor_validator = SensorValidator()
