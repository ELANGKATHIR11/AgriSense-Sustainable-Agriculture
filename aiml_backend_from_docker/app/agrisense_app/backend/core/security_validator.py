"""
Security Validation and Input Protection
Implements AI safety, sensor validation, and rate limiting
Part of AgriSense Production Optimization Blueprint
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ============================================================================
# Sensor Value Validation
# ============================================================================

class SensorValidationError(Exception):
    """Raised when sensor data is invalid or suspicious"""
    pass


class SensorReading(BaseModel):
    """
    Validated sensor reading model with strict bounds checking
    """
    device_id: str = Field(..., min_length=1, max_length=100)
    temperature: float = Field(..., ge=-50, le=70, description="Temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Humidity in %")
    soil_moisture: float = Field(..., ge=0, le=100, description="Soil moisture in %")
    ph_level: Optional[float] = Field(None, ge=0, le=14, description="pH level")
    nitrogen: Optional[float] = Field(None, ge=0, le=200, description="N in ppm")
    phosphorus: Optional[float] = Field(None, ge=0, le=200, description="P in ppm")
    potassium: Optional[float] = Field(None, ge=0, le=200, description="K in ppm")
    timestamp: Optional[datetime] = None
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """Additional temperature validation"""
        if v < -20 or v > 60:
            logger.warning(f"Suspicious temperature value: {v}°C")
        return v
    
    @validator('humidity')
    def validate_humidity(cls, v):
        """Additional humidity validation"""
        if v > 95:
            logger.warning(f"Suspicious humidity value: {v}%")
        return v
    
    @validator('soil_moisture')
    def validate_soil_moisture(cls, v):
        """Additional soil moisture validation"""
        if v < 5 or v > 90:
            logger.warning(f"Suspicious soil moisture value: {v}%")
        return v


def validate_sensor_reading(reading: Dict[str, Any]) -> SensorReading:
    """
    Validate sensor reading and detect spoofing attempts
    
    Args:
        reading: Raw sensor reading dictionary
        
    Returns:
        Validated SensorReading object
        
    Raises:
        SensorValidationError: If validation fails
        HTTPException: If data is malicious
    """
    try:
        validated = SensorReading(**reading)
        
        # Additional anomaly detection
        detect_sensor_spoofing(validated)
        
        return validated
    except Exception as e:
        logger.error(f"Sensor validation failed: {e}")
        raise SensorValidationError(f"Invalid sensor data: {e}")


def detect_sensor_spoofing(reading: Optional[SensorReading] = None, **kwargs):
    """
    Detect impossible or malicious sensor values.

    Accepts either a `SensorReading` instance, a dict-like `reading`, or
    keyword arguments containing temperature/humidity/soil_moisture.

    Raises:
        HTTPException: If spoofing detected
    """
    # Called with a SensorReading -> raise on spoof detection (validation flow)
    if isinstance(reading, SensorReading):
        temperature = getattr(reading, "temperature", None)
        humidity = getattr(reading, "humidity", None)

        try:
            if temperature is not None and humidity is not None:
                if (temperature > 50 and humidity > 80) or (temperature > 30 and humidity < 20):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Sensor spoofing detected: Impossible temperature/humidity combination"
                    )
        except TypeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sensor spoofing detected: Invalid sensor value types"
            )

        return False

    # Called as a boolean check with kwargs or dict; return True/False
    if isinstance(reading, dict):
        temperature = reading.get("temperature")
        humidity = reading.get("humidity")
    else:
        temperature = kwargs.get("temperature")
        humidity = kwargs.get("humidity")

    try:
        if temperature is not None and humidity is not None:
            return (temperature > 50 and humidity > 80) or (temperature > 30 and humidity < 20)
    except TypeError:
        return True

    return False


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter
    Protects ML endpoints from abuse
    """
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.rate = requests_per_minute / 60.0  # Tokens per second
        self.burst_size = burst_size
        self.tokens: Dict[str, float] = defaultdict(lambda: burst_size)
        self.last_update: Dict[str, float] = defaultdict(lambda: time.time())
        self.lock = Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client
        
        Args:
            client_id: Unique client identifier (IP, user_id, etc.)
            
        Returns:
            True if request is allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            
            # Add tokens based on time elapsed
            time_passed = now - self.last_update[client_id]
            self.tokens[client_id] = min(
                self.burst_size,
                self.tokens[client_id] + time_passed * self.rate
            )
            self.last_update[client_id] = now
            
            # Check if we have tokens available
            if self.tokens[client_id] >= 1:
                self.tokens[client_id] -= 1
                return True
            else:
                return False
    
    def get_wait_time(self, client_id: str) -> float:
        """Get seconds until next token is available"""
        with self.lock:
            if self.tokens[client_id] >= 1:
                return 0
            return (1 - self.tokens[client_id]) / self.rate


# Global rate limiters for different endpoint types
_ml_rate_limiter = RateLimiter(requests_per_minute=30, burst_size=5)
_api_rate_limiter = RateLimiter(requests_per_minute=120, burst_size=20)
_sensor_rate_limiter = RateLimiter(requests_per_minute=600, burst_size=100)  # IoT devices


async def check_rate_limit(
    request: Request,
    limiter: RateLimiter = _api_rate_limiter
):
    """
    FastAPI dependency for rate limiting
    
    Usage:
        @app.post("/predict", dependencies=[Depends(rate_limit_ml)])
        async def predict():
            ...
    """
    # Use IP address as client ID (or use user_id if authenticated)
    client_id = request.client.host if request.client else "unknown"
    
    if not limiter.is_allowed(client_id):
        wait_time = limiter.get_wait_time(client_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {wait_time:.1f} seconds",
            headers={"Retry-After": str(int(wait_time) + 1)}
        )


# Predefined rate limit dependencies
async def rate_limit_api(request: Request):
    """Standard API rate limit (120/min)"""
    await check_rate_limit(request, _api_rate_limiter)


async def rate_limit_ml(request: Request):
    """ML endpoint rate limit (30/min)"""
    await check_rate_limit(request, _ml_rate_limiter)


async def rate_limit_sensor(request: Request):
    """Sensor endpoint rate limit (600/min)"""
    await check_rate_limit(request, _sensor_rate_limiter)


# ============================================================================
# AI Safety - Input Validation
# ============================================================================

def validate_ml_input(features: Dict[str, float]) -> Dict[str, float]:
    """
    Validate ML model input features
    
    Args:
        features: Input feature dictionary
        
    Returns:
        Validated features
        
    Raises:
        HTTPException: If input is invalid
    """
    required_features = ["temperature", "humidity", "soil_moisture"]
    
    # Check required features
    for feature in required_features:
        if feature not in features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required feature: {feature}"
            )
    
    # Validate ranges
    try:
        reading = SensorReading(**features)
        return features
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input features: {e}"
        )


def sanitize_text_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize text input for chatbot/search queries
    
    Args:
        text: User input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
        
    Raises:
        HTTPException: If input is too long or contains malicious content
    """
    if len(text) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input too long (max {max_length} characters)"
        )

    # Remove potential injection attempts (sanitize rather than reject)
    dangerous_patterns = [
        "<script", "javascript:", "onerror=", "onload=",
        "<?php", "<?", "<%", "${", "exec(", "eval("
    ]

    import re

    sanitized = text
    for pattern in dangerous_patterns:
        # Remove occurrences of the pattern case-insensitively
        try:
            sanitized = re.sub(re.escape(pattern), "", sanitized, flags=re.IGNORECASE)
        except re.error:
            # Fallback: simple replace
            sanitized = sanitized.replace(pattern, "")

    # Also strip common angle-bracketed tags as a best-effort
    sanitized = re.sub(r"<.*?>", "", sanitized)

    return sanitized.strip()


# ============================================================================
# Request Validation Middleware
# ============================================================================

class SecurityValidationMiddleware:
    """
    Middleware to validate all incoming requests
    """
    
    def __init__(self, app):
        self.app = app
        self.suspicious_ips: Dict[str, int] = defaultdict(int)
        self.suspicious_ip_lock = Lock()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Get client IP
        client_ip = None
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"x-forwarded-for":
                client_ip = header_value.decode().split(",")[0].strip()
                break
        
        if not client_ip and scope.get("client"):
            client_ip = scope["client"][0]
        
        # Check if IP is blocked
        if client_ip and self.is_ip_blocked(client_ip):
            # Send 403 Forbidden
            response = {
                "type": "http.response.start",
                "status": 403,
                "headers": [(b"content-type", b"application/json")],
            }
            await send(response)
            
            body = {
                "type": "http.response.body",
                "body": b'{"detail":"IP address blocked due to suspicious activity"}',
            }
            await send(body)
            return
        
        await self.app(scope, receive, send)
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        with self.suspicious_ip_lock:
            return self.suspicious_ips.get(ip, 0) >= 100
    
    def report_suspicious_activity(self, ip: str):
        """Report suspicious activity from IP"""
        with self.suspicious_ip_lock:
            self.suspicious_ips[ip] += 1
            if self.suspicious_ips[ip] == 100:
                logger.error(f"🚨 Blocked IP due to suspicious activity: {ip}")


# ============================================================================
# Device Security
# ============================================================================

class DeviceCertificateValidator:
    """
    Validate per-device certificates for ESP32/IoT devices
    Implement this with actual certificate validation in production
    """
    
    def __init__(self):
        self.valid_device_ids: set = set()
        # In production, load from secure database
    
    def validate_device_certificate(
        self,
        device_id: str,
        certificate: Optional[str] = None
    ) -> bool:
        """
        Validate device certificate
        
        Args:
            device_id: Device identifier
            certificate: Device certificate (optional for now)
            
        Returns:
            True if valid, False otherwise
        """
        # TODO: Implement actual certificate validation
        # For now, just check if device is registered
        return True  # Placeholder
    
    def register_device(self, device_id: str):
        """Register new device"""
        self.valid_device_ids.add(device_id)
        logger.info(f"✅ Registered device: {device_id}")
    
    def revoke_device(self, device_id: str):
        """Revoke device certificate"""
        self.valid_device_ids.discard(device_id)
        logger.warning(f"⚠️ Revoked device: {device_id}")


# Global device validator
_device_validator = DeviceCertificateValidator()


# ============================================================================
# Example Usage
# ============================================================================

"""
# In your main.py:

from agrisense_app.backend.core.security_validator import (
    validate_sensor_reading,
    rate_limit_ml,
    rate_limit_sensor,
    validate_ml_input,
    sanitize_text_input,
    SecurityValidationMiddleware
)

# Add middleware
app.add_middleware(SecurityValidationMiddleware)

# Use in routes
@app.post("/ingest", dependencies=[Depends(rate_limit_sensor)])
async def ingest_sensor_data(data: dict):
    validated_reading = validate_sensor_reading(data)
    # Process validated data
    ...

@app.post("/predict", dependencies=[Depends(rate_limit_ml)])
async def predict(features: dict):
    validated_features = validate_ml_input(features)
    # Run ML model
    ...

@app.post("/chatbot/ask")
async def ask_chatbot(question: str):
    clean_question = sanitize_text_input(question, max_length=500)
    # Process chatbot query
    ...
"""


if __name__ == "__main__":
    # Test validation functions
    
    # Test sensor validation
    valid_reading = {
        "device_id": "ESP32_001",
        "temperature": 25.5,
        "humidity": 65.0,
        "soil_moisture": 42.0,
        "ph_level": 6.5
    }
    
    try:
        validated = validate_sensor_reading(valid_reading)
        print(f"✅ Valid reading: {validated.device_id}")
    except SensorValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    # Test rate limiting
    limiter = RateLimiter(requests_per_minute=60)
    
    # Simulate requests
    for i in range(70):
        if limiter.is_allowed("test_client"):
            print(f"Request {i+1}: ✅ Allowed")
        else:
            wait = limiter.get_wait_time("test_client")
            print(f"Request {i+1}: ❌ Rate limited (wait {wait:.2f}s)")
