"""
Structured JSON Logging for AgriSense
Production-ready logging with sampling and metrics tracking
"""
import json
import logging
import sys
import random
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar

from ..config.optimization import settings

# Context variable for request ID tracking
request_id_ctx: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging.
    Outputs logs in JSON format for easy parsing by log aggregators.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add request ID if available
        request_id = request_id_ctx.get()
        if request_id:
            log_entry["request_id"] = request_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add custom fields from extra
        if hasattr(record, 'custom_fields'):
            log_entry.update(record.custom_fields)
        
        return json.dumps(log_entry)


class SamplingHandler(logging.Handler):
    """
    Logging handler that samples logs to reduce noise in production.
    Only logs a percentage of messages based on sampling rate.
    """
    
    def __init__(self, base_handler: logging.Handler, sampling_rate: float = 0.1):
        super().__init__()
        self.base_handler = base_handler
        self.sampling_rate = sampling_rate
    
    def emit(self, record: logging.LogRecord):
        """Emit log record with sampling"""
        # Always log ERROR and CRITICAL
        if record.levelno >= logging.ERROR:
            self.base_handler.emit(record)
            return
        
        # Sample other levels
        if random.random() < self.sampling_rate:
            self.base_handler.emit(record)


def setup_logging():
    """
    Configure logging for AgriSense.
    Sets up structured logging with optional sampling.
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Use structured formatter if enabled
    if settings.enable_structured_logging:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    
    # Wrap with sampling handler if enabled
    if settings.enable_log_sampling and settings.log_sampling_rate < 1.0:
        handler = SamplingHandler(console_handler, settings.log_sampling_rate)
    else:
        handler = console_handler
    
    root_logger.addHandler(handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    logging.info("Logging configured", extra={
        "custom_fields": {
            "structured": settings.enable_structured_logging,
            "sampling_enabled": settings.enable_log_sampling,
            "sampling_rate": settings.log_sampling_rate
        }
    })


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with structured logging support.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Processing sensor data", extra={
            "custom_fields": {
                "device_id": "ESP32_001",
                "sensor_type": "temperature"
            }
        })
    """
    return logging.getLogger(name)


# ===== CONVENIENCE LOGGING FUNCTIONS =====

def log_ml_prediction(
    model_name: str,
    input_data: Dict[str, Any],
    prediction: Any,
    confidence: float,
    inference_time_ms: float
):
    """Log ML prediction with metrics"""
    logger = get_logger("agrisense.ml")
    logger.info(
        f"ML prediction: {model_name}",
        extra={
            "custom_fields": {
                "model": model_name,
                "prediction": str(prediction),
                "confidence": confidence,
                "inference_time_ms": inference_time_ms,
                "input_features": len(input_data)
            }
        }
    )


def log_sensor_reading(
    device_id: str,
    sensor_type: str,
    value: float,
    unit: str
):
    """Log sensor reading"""
    logger = get_logger("agrisense.sensors")
    logger.debug(
        f"Sensor reading: {device_id}",
        extra={
            "custom_fields": {
                "device_id": device_id,
                "sensor_type": sensor_type,
                "value": value,
                "unit": unit
            }
        }
    )


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None
):
    """Log API request with metrics"""
    logger = get_logger("agrisense.api")
    logger.info(
        f"{method} {path} {status_code}",
        extra={
            "custom_fields": {
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "user_id": user_id
            }
        }
    )


def log_error(
    component: str,
    error_type: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None
):
    """Log error with context"""
    logger = get_logger(f"agrisense.{component}")
    logger.error(
        f"{error_type}: {error_message}",
        extra={
            "custom_fields": {
                "component": component,
                "error_type": error_type,
                "context": context or {}
            }
        }
    )


def log_security_event(
    event_type: str,
    user: Optional[str],
    action: str,
    result: str,
    details: Optional[Dict[str, Any]] = None
):
    """Log security-related events"""
    logger = get_logger("agrisense.security")
    logger.warning(
        f"Security event: {event_type}",
        extra={
            "custom_fields": {
                "event_type": event_type,
                "user": user,
                "action": action,
                "result": result,
                "details": details or {}
            }
        }
    )
