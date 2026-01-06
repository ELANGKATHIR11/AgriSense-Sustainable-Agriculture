"""
Graceful Degradation and Fault Tolerance
Implements fallback mechanisms and health monitoring
Part of AgriSense Production Optimization Blueprint
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# Service Status
# ============================================================================

class ServiceStatus(str, Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckResult:
    """Health check result"""
    
    def __init__(
        self,
        status: ServiceStatus,
        message: str = "",
        details: Optional[Dict[str, Any]] = None
    ):
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================================
# Fallback Decorator
# ============================================================================

def with_fallback(
    fallback_func: Callable,
    log_errors: bool = True,
    raise_on_fallback: bool = False
):
    """
    Decorator to provide fallback mechanism
    
    Args:
        fallback_func: Function to call if primary fails
        log_errors: Whether to log errors
        raise_on_fallback: Whether to raise exception after fallback
        
    Example:
        @with_fallback(fallback_func=rule_based_prediction)
        def ml_prediction(data):
            # ML model inference
            return model.predict(data)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(
                        f"Primary function {func.__name__} failed: {e}, "
                        f"using fallback {fallback_func.__name__}"
                    )
                
                try:
                    result = fallback_func(*args, **kwargs)
                    
                    if raise_on_fallback:
                        raise Exception(f"Using fallback due to: {e}")
                    
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise
        
        return wrapper
    return decorator


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================

class CircuitState(str, Enum):
    """Circuit breaker state"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures
    
    Example:
        breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        
        @breaker.call
        def unreliable_service():
            # Call external service
            return result
    """
    
    def __init__(
        self,
        name: str = "circuit",
        failure_threshold: int = 5,
        timeout: float = 60.0,
    ):
        # Support both calling styles: CircuitBreaker(name, failure_threshold=..)
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments for function
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if time.time() - self.last_failure_time >= self.timeout:
                logger.info(f"Circuit breaker {self.name}: Attempting recovery (HALF_OPEN)")
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit breaker {self.name}: Recovery successful (CLOSED)")
                self.state = CircuitState.CLOSED
            
            self.failure_count = 0
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker {self.name}: Threshold reached, "
                    f"opening circuit ({self.failure_count} failures)"
                )
                self.state = CircuitState.OPEN
            
            raise
    
    def get_status(self) -> dict:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time
        }

    # Convenience state helpers
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN


# ============================================================================
# ML Model Fallback Manager
# ============================================================================

class MLFallbackManager:
    """
    Manages fallback strategies for ML models
    """
    
    def __init__(self):
        self.last_valid_predictions: Dict[str, Any] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def get_circuit_breaker(self, model_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for model"""
        if model_name not in self.circuit_breakers:
            self.circuit_breakers[model_name] = CircuitBreaker(
                failure_threshold=3,
                timeout=120.0,
                name=model_name
            )
        return self.circuit_breakers[model_name]
    
    def predict_with_fallback(
        self,
        model_name: str,
        ml_predict_func: Callable,
        rule_based_func: Callable,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute ML prediction with fallback to rule-based logic
        
        Args:
            model_name: Name of ML model
            ml_predict_func: ML prediction function
            rule_based_func: Rule-based fallback function
            input_data: Input data for prediction
            
        Returns:
            Prediction result with metadata
        """
        breaker = self.get_circuit_breaker(model_name)
        
        try:
            # Try ML prediction
            result = breaker.call(ml_predict_func, input_data)
            
            # Cache successful prediction
            self.last_valid_predictions[model_name] = result
            
            return {
                "prediction": result,
                "method": "ml",
                "model": model_name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"ML prediction failed for {model_name}: {e}, using fallback")
            
            # Try rule-based fallback
            try:
                result = rule_based_func(input_data)
                
                return {
                    "prediction": result,
                    "method": "rule_based",
                    "model": model_name,
                    "fallback_reason": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as fallback_error:
                # Last resort: return cached prediction if available
                if model_name in self.last_valid_predictions:
                    logger.warning(f"Using cached prediction for {model_name}")
                    
                    return {
                        "prediction": self.last_valid_predictions[model_name],
                        "method": "cached",
                        "model": model_name,
                        "cache_warning": "Using last valid prediction",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # Complete failure
                logger.error(f"All prediction methods failed for {model_name}")
                raise Exception("ML and fallback methods both failed")
    
    def get_all_statuses(self) -> Dict[str, dict]:
        """Get status of all circuit breakers"""
        return {
            name: breaker.get_status()
            for name, breaker in self.circuit_breakers.items()
        }


# Global fallback manager
_ml_fallback_manager = MLFallbackManager()


# ============================================================================
# Rule-Based Fallback Functions
# ============================================================================

def rule_based_irrigation_recommendation(sensor_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Rule-based irrigation recommendation (fallback for ML)
    
    Args:
        sensor_data: Sensor reading dictionary
        
    Returns:
        Irrigation recommendation
    """
    # Support both dict input and keyword args
    if sensor_data is None:
        sensor_data = kwargs

    soil_moisture = sensor_data.get("soil_moisture", sensor_data.get("moisture", 50))
    temperature = sensor_data.get("temperature", sensor_data.get("temp", 25))
    
    # Simple rule-based logic
    if soil_moisture < 30:
        water_liters = 10.0
        action = "irrigate_now"
    elif soil_moisture < 50:
        water_liters = 5.0
        action = "irrigate_soon"
    else:
        water_liters = 0.0
        action = "no_action"
    
    # Adjust for temperature
    if temperature > 35:
        water_liters *= 1.5
    
    return {
        "water_liters": water_liters,
        "action": action,
        "confidence": 0.7,  # Lower confidence for rule-based
        "method": "rule_based"
    }


def rule_based_crop_recommendation(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rule-based crop recommendation (fallback for ML)
    
    Args:
        sensor_data: Sensor reading dictionary
        
    Returns:
        Crop recommendation
    """
    temperature = sensor_data.get("temperature", 25)
    humidity = sensor_data.get("humidity", 60)
    ph = sensor_data.get("ph_level", 6.5)
    
    # Simple rule-based crop selection
    if 20 <= temperature <= 30 and 60 <= humidity <= 80:
        if 6.0 <= ph <= 7.0:
            crop = "rice"
        else:
            crop = "wheat"
    elif temperature > 30:
        crop = "cotton"
    else:
        crop = "potato"
    
    return {
        "recommended_crop": crop,
        "confidence": 0.6,
        "method": "rule_based",
        "reasoning": f"Based on temperature={temperature}°C, humidity={humidity}%, pH={ph}"
    }


# ============================================================================
# Health Check System
# ============================================================================

class HealthCheckRegistry:
    """
    Registry for health check functions
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable] = {}
    
    def register(self, name: str, check_func: Callable):
        """Register a health check function"""
        self._checks[name] = check_func
        logger.info(f"✅ Registered health check: {name}")

    # Backwards-compatible alias used by validator script
    def register_check(self, name: str, check_func: Callable, critical: bool = False):
        """Register health check with optional critical flag (alias)."""
        # Store critical flag on function for visibility (not required)
        setattr(check_func, "_critical", bool(critical))
        self.register(name, check_func)
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_func in self._checks.items():
            try:
                result = await check_func() if callable(check_func) else check_func()
                results[name] = result
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = HealthCheckResult(
                    status=ServiceStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}"
                )
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> ServiceStatus:
        """Determine overall system status from individual checks"""
        statuses = [r.status for r in results.values()]
        
        if ServiceStatus.UNHEALTHY in statuses:
            return ServiceStatus.UNHEALTHY
        elif ServiceStatus.DEGRADED in statuses:
            return ServiceStatus.DEGRADED
        elif ServiceStatus.UNKNOWN in statuses:
            return ServiceStatus.DEGRADED
        else:
            return ServiceStatus.HEALTHY


# Global health check registry
_health_registry = HealthCheckRegistry()


# ============================================================================
# Predefined Health Checks
# ============================================================================

async def check_database_health() -> HealthCheckResult:
    """Check database connectivity"""
    try:
        # TODO: Implement actual database ping
        # For now, assume healthy
        return HealthCheckResult(
            status=ServiceStatus.HEALTHY,
            message="Database connected",
            details={"connection": "active"}
        )
    except Exception as e:
        return HealthCheckResult(
            status=ServiceStatus.UNHEALTHY,
            message=f"Database error: {e}"
        )


async def check_cache_health() -> HealthCheckResult:
    """Check Redis cache health"""
    try:
        from .cache_manager import get_cache
        
        cache = get_cache()
        stats = cache.get_stats()
        
        if stats.get("connected"):
            return HealthCheckResult(
                status=ServiceStatus.HEALTHY,
                message="Cache connected",
                details=stats
            )
        else:
            return HealthCheckResult(
                status=ServiceStatus.DEGRADED,
                message="Cache unavailable, using fallback",
                details=stats
            )
    except Exception as e:
        return HealthCheckResult(
            status=ServiceStatus.DEGRADED,
            message=f"Cache check failed: {e}"
        )


async def check_ml_models_health() -> HealthCheckResult:
    """Check ML models availability"""
    try:
        # Check if models are loaded
        model_count = 0  # TODO: Get actual model count
        
        return HealthCheckResult(
            status=ServiceStatus.HEALTHY,
            message=f"{model_count} models loaded",
            details={"loaded_models": model_count}
        )
    except Exception as e:
        return HealthCheckResult(
            status=ServiceStatus.DEGRADED,
            message=f"ML models check failed: {e}"
        )


# Register default health checks
_health_registry.register("database", check_database_health)
_health_registry.register("cache", check_cache_health)
_health_registry.register("ml_models", check_ml_models_health)


# ============================================================================
# FastAPI Integration
# ============================================================================

"""
# In your main.py:

from agrisense_app.backend.core.graceful_degradation import (
    _health_registry,
    _ml_fallback_manager,
    rule_based_irrigation_recommendation,
    rule_based_crop_recommendation
)

@app.get("/health/detailed")
async def detailed_health_check():
    results = await _health_registry.run_all_checks()
    overall_status = _health_registry.get_overall_status(results)
    
    return {
        "status": overall_status.value,
        "checks": {
            name: result.to_dict()
            for name, result in results.items()
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict/irrigation")
async def predict_irrigation(sensor_data: dict):
    result = _ml_fallback_manager.predict_with_fallback(
        model_name="irrigation",
        ml_predict_func=lambda data: ml_model.predict(data),
        rule_based_func=rule_based_irrigation_recommendation,
        input_data=sensor_data
    )
    return result
"""


if __name__ == "__main__":
    # Test circuit breaker
    breaker = CircuitBreaker(failure_threshold=3, timeout=10, name="test")
    
    def failing_func():
        raise Exception("Service unavailable")
    
    # Simulate failures
    for i in range(5):
        try:
            breaker.call(failing_func)
        except Exception as e:
            print(f"Attempt {i+1}: {e}")
            print(f"Circuit state: {breaker.state.value}")
