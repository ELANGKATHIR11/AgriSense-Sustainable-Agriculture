"""
Structured Logging and Observability
Implements JSON logging, metrics collection, and monitoring
Part of AgriSense Production Optimization Blueprint
"""

import json
import logging
import os
import sys
import time
import traceback
from contextvars import ContextVar
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

# Request ID context for tracing
request_id_ctx: ContextVar[str] = ContextVar('request_id', default='')


# ============================================================================
# Log Levels and Sampling
# ============================================================================

class LogLevel(str, Enum):
    """Log level enum"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Sampling configuration
LOG_SAMPLING_RATE = float(os.getenv("LOG_SAMPLING_RATE", "1.0"))  # 1.0 = no sampling
ENABLE_DEBUG_LOGS = os.getenv("ENABLE_DEBUG_LOGS", "false").lower() == "true"


def should_sample_log() -> bool:
    """Determine if log should be sampled"""
    import random
    return random.random() < LOG_SAMPLING_RATE


# ============================================================================
# Structured JSON Logger
# ============================================================================

class StructuredFormatter(logging.Formatter):
    """
    Custom log formatter that outputs structured JSON logs
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname.lower(),
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
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        if hasattr(record, 'extra_data'):
            log_entry["extra"] = record.extra_data
        
        # Add custom fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                           'levelname', 'levelno', 'lineno', 'module', 'msecs',
                           'message', 'pathname', 'process', 'processName',
                           'relativeCreated', 'thread', 'threadName', 'exc_info',
                           'exc_text', 'stack_info', 'extra_data']:
                if not key.startswith('_'):
                    log_entry[key] = value
        
        return json.dumps(log_entry)


class SampledStructuredFormatter(StructuredFormatter):
    """Structured formatter with sampling for high-volume logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with sampling"""
        # Always log errors and warnings
        if record.levelno >= logging.WARNING:
            return super().format(record)
        
        # Sample other logs
        if should_sample_log():
            return super().format(record)
        
        # Skipped logs (return empty to filter out)
        return ""


def setup_structured_logging(
    level: str = "INFO",
    enable_sampling: bool = False
):
    """
    Configure structured JSON logging for the application
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_sampling: Enable log sampling to reduce noise
        
    Usage:
        # In main.py startup
        setup_structured_logging(level="INFO", enable_sampling=True)
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Use sampled formatter if enabled
    if enable_sampling:
        formatter = SampledStructuredFormatter()
    else:
        formatter = StructuredFormatter()
    
    handler.setFormatter(formatter)
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    handler.setLevel(log_level)
    
    root_logger.addHandler(handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    
    logging.info(
        "Structured logging configured",
        extra={"log_level": level, "sampling_enabled": enable_sampling}
    )


# ============================================================================
# Enhanced Logger with Extra Context
# ============================================================================

class ContextLogger:
    """
    Logger wrapper that adds contextual information
    
    Example:
        logger = ContextLogger("agrisense.api")
        logger.info("Processing request", user_id="123", device_id="ESP32_001")
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with context"""
        extra_data = kwargs
        self.logger.log(level, message, extra={'extra_data': extra_data})
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        if ENABLE_DEBUG_LOGS:
            self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        extra_data = kwargs
        self.logger.exception(message, extra={'extra_data': extra_data})


# ============================================================================
# Request Logging Middleware
# ============================================================================

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests with timing and status
    """
    
    def __init__(self, app, logger: Optional[ContextLogger] = None):
        super().__init__(app)
        self.logger = logger or ContextLogger("agrisense.requests")
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log details"""
        
        # Generate request ID
        request_id = str(uuid4())
        request_id_ctx.set(request_id)
        
        # Start timing
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log successful request
            self.logger.info(
                "HTTP request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
                request_id=request_id,
                client_ip=request.client.host if request.client else None
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log failed request
            self.logger.error(
                "HTTP request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration * 1000, 2),
                request_id=request_id,
                client_ip=request.client.host if request.client else None
            )
            
            raise


# ============================================================================
# Metrics Collection
# ============================================================================

class MetricsCollector:
    """
    Lightweight metrics collector for monitoring
    """
    
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list] = {}
        self.start_time = time.time()
    
    def increment(self, metric: str, value: int = 1):
        """Increment a counter"""
        # Support Enum-like metrics by using .value when present
        metric_key = getattr(metric, "value", metric)
        self.counters[metric_key] = self.counters.get(metric_key, 0) + value
    
    def set_gauge(self, metric: str, value: float):
        """Set a gauge value"""
        metric_key = getattr(metric, "value", metric)
        self.gauges[metric_key] = value
    
    def record(self, metric: str, value: float, max_size: int = 1000):
        """Record a value in histogram"""
        metric_key = getattr(metric, "value", metric)
        if metric_key not in self.histograms:
            self.histograms[metric_key] = []

        self.histograms[metric_key].append(value)
        
        # Keep only recent values
        if len(self.histograms[metric]) > max_size:
            self.histograms[metric] = self.histograms[metric][-max_size:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all metrics as dictionary"""
        # Coerce any metric keys to strings to avoid exposing internal objects
        coerced_counters = {}
        for k, v in self.counters.items():
            try:
                key = getattr(k, "value", k)
            except Exception:
                key = k
            coerced_counters[str(key)] = v

        coerced_gauges = {}
        for k, v in self.gauges.items():
            try:
                key = getattr(k, "value", k)
            except Exception:
                key = k
            coerced_gauges[str(key)] = v

        stats = {
            "uptime_seconds": time.time() - self.start_time,
            "counters": coerced_counters,
            "gauges": coerced_gauges,
        }
        
        # Calculate histogram statistics
        histogram_stats = {}
        for name, values in self.histograms.items():
            if values:
                sorted_values = sorted(values)
                n = len(sorted_values)
                
                histogram_stats[name] = {
                    "count": n,
                    "min": sorted_values[0],
                    "max": sorted_values[-1],
                    "mean": sum(sorted_values) / n,
                    "p50": sorted_values[int(n * 0.5)],
                    "p95": sorted_values[int(n * 0.95)],
                    "p99": sorted_values[int(n * 0.99)],
                }
        
        stats["histograms"] = histogram_stats

        # Backwards-compatible: expose individual metric keys at top-level
        # so code expecting flat metric keys can find them easily.
        try:
            for name, count in coerced_counters.items():
                if name not in stats:
                    stats[name] = count
        except Exception:
            pass

        try:
            for name, value in coerced_gauges.items():
                if name not in stats:
                    stats[name] = value
        except Exception:
            pass

        return stats


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector"""
    return _metrics


# ============================================================================
# Tracked Metrics for AgriSense
# ============================================================================

class _Metric:
    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return self.value


class AgriSenseMetrics:
    """
    Predefined metrics for AgriSense application (metric objects expose `.value`).
    """
    # API Metrics
    HTTP_REQUESTS_TOTAL = _Metric("http.requests.total")
    HTTP_REQUESTS_ERRORS = _Metric("http.requests.errors")
    HTTP_REQUEST_DURATION = _Metric("http.request.duration_ms")
    # Backwards-compatible aliases
    API_REQUESTS_TOTAL = HTTP_REQUESTS_TOTAL
    API_REQUESTS_ERRORS = HTTP_REQUESTS_ERRORS
    API_REQUEST_DURATION = HTTP_REQUEST_DURATION

    # ML Metrics
    ML_PREDICTIONS_TOTAL = _Metric("ml.predictions.total")
    ML_PREDICTION_DURATION = _Metric("ml.prediction.duration_ms")
    ML_CONFIDENCE_SCORE = _Metric("ml.confidence.score")
    ML_FALLBACK_COUNT = _Metric("ml.fallback.count")

    # Sensor Metrics
    SENSOR_READINGS_TOTAL = _Metric("sensor.readings.total")
    SENSOR_VALIDATION_ERRORS = _Metric("sensor.validation_errors")
    SENSOR_DRIFT_DETECTED = _Metric("sensor.drift.detected")

    # Cache Metrics
    CACHE_HITS = _Metric("cache.hits")
    CACHE_MISSES = _Metric("cache.misses")

    # Alert Metrics
    ALERTS_SENT = _Metric("alerts.sent")
    ALERT_FALSE_POSITIVES = _Metric("alerts.false_positives")

    # Water Efficiency
    WATER_SAVED_LITERS = _Metric("water.saved.liters")
    WATER_EFFICIENCY_PERCENT = _Metric("water.efficiency.percent")


# ============================================================================
# Example Usage
# ============================================================================

"""
# In main.py:

from agrisense_app.backend.core.observability import (
    setup_structured_logging,
    RequestLoggingMiddleware,
    ContextLogger,
    get_metrics,
    AgriSenseMetrics
)

# Setup logging
setup_structured_logging(level="INFO", enable_sampling=True)

# Create logger
logger = ContextLogger("agrisense.main")

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Get metrics collector
metrics = get_metrics()

# In your route handlers:

@app.post("/ingest")
async def ingest_sensor_data(data: dict):
    # Increment counter
    metrics.increment(AgriSenseMetrics.SENSOR_READINGS_TOTAL)
    
    # Log with context
    logger.info(
        "Sensor data ingested",
        device_id=data.get("device_id"),
        temperature=data.get("temperature")
    )
    
    # Process data...
    return {"status": "ok"}

@app.post("/predict")
async def predict(features: dict):
    start_time = time.time()
    
    try:
        result = model.predict(features)
        
        # Record metrics
        duration = (time.time() - start_time) * 1000
        metrics.increment(AgriSenseMetrics.ML_PREDICTIONS_TOTAL)
        metrics.record(AgriSenseMetrics.ML_PREDICTION_DURATION, duration)
        
        logger.info(
            "ML prediction completed",
            duration_ms=round(duration, 2),
            confidence=result.get("confidence")
        )
        
        return result
        
    except Exception as e:
        logger.error("ML prediction failed", error=str(e))
        raise

@app.get("/metrics")
async def metrics_endpoint():
    return get_metrics().get_stats()
"""


if __name__ == "__main__":
    # Test structured logging
    setup_structured_logging(level="INFO", enable_sampling=False)
    
    logger = ContextLogger("test")
    
    logger.info("Test info message", user_id="123", action="login")
    logger.warning("Test warning", temperature=45.5, threshold=40.0)
    
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.exception("Test exception logging", context="test")
    
    # Test metrics
    metrics = get_metrics()
    metrics.increment(AgriSenseMetrics.HTTP_REQUESTS_TOTAL)
    metrics.record(AgriSenseMetrics.ML_PREDICTION_DURATION, 125.5)
    metrics.set_gauge("cpu_usage", 45.2)
    
    print("\nMetrics:")
    print(json.dumps(metrics.get_stats(), indent=2))
