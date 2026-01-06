"""
Graceful Degradation and Fault Tolerance
Fallback mechanisms when ML models or services fail
"""
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
import traceback

from ..config.optimization import settings

logger = logging.getLogger(__name__)


class MLFailureError(Exception):
    """Raised when ML prediction fails"""
    pass


class FallbackManager:
    """
    Manages fallback strategies for ML failures.
    Provides rule-based alternatives when ML is unavailable.
    """
    
    def __init__(self):
        self.last_successful_predictions: Dict[str, Any] = {}
        self.failure_counts: Dict[str, int] = {}
        
    def with_fallback(self, fallback_func: Optional[Callable] = None):
        """
        Decorator to add fallback logic to ML functions.
        
        Usage:
            @fallback_manager.with_fallback(fallback_func=rule_based_recommendation)
            async def ml_recommendation(data):
                return model.predict(data)
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                func_name = func.__name__
                
                if not settings.enable_graceful_degradation:
                    # No fallback, just execute
                    return await func(*args, **kwargs)
                
                try:
                    # Try ML prediction
                    result = await func(*args, **kwargs)
                    
                    # Cache successful prediction
                    if settings.cache_last_prediction:
                        cache_key = f"{func_name}:{str(args)}:{str(kwargs)}"
                        self.last_successful_predictions[cache_key] = result
                    
                    # Reset failure count on success
                    self.failure_counts[func_name] = 0
                    
                    return result
                
                except Exception as e:
                    logger.error(f"ML function {func_name} failed: {e}")
                    logger.debug(traceback.format_exc())
                    
                    # Increment failure count
                    self.failure_counts[func_name] = self.failure_counts.get(func_name, 0) + 1
                    
                    # Try fallback strategies in order
                    
                    # 1. Use cached prediction if available
                    if settings.cache_last_prediction:
                        cache_key = f"{func_name}:{str(args)}:{str(kwargs)}"
                        if cache_key in self.last_successful_predictions:
                            logger.info(f"Using cached prediction for {func_name}")
                            return self.last_successful_predictions[cache_key]
                    
                    # 2. Use rule-based fallback
                    if settings.ml_fallback_to_rules and fallback_func:
                        logger.info(f"Using rule-based fallback for {func_name}")
                        try:
                            return await fallback_func(*args, **kwargs)
                        except Exception as fallback_error:
                            logger.error(f"Fallback also failed: {fallback_error}")
                    
                    # 3. All fallbacks failed
                    logger.error(
                        f"All fallback strategies exhausted for {func_name}. "
                        f"Failure count: {self.failure_counts[func_name]}"
                    )
                    raise MLFailureError(
                        f"ML prediction and all fallbacks failed for {func_name}"
                    )
            
            return wrapper
        return decorator
    
    def get_failure_stats(self) -> Dict[str, int]:
        """Get failure statistics for monitoring"""
        return self.failure_counts.copy()
    
    def reset_stats(self):
        """Reset failure statistics"""
        self.failure_counts.clear()


# Singleton instance
fallback_manager = FallbackManager()


# ===== RULE-BASED FALLBACK FUNCTIONS =====

async def rule_based_crop_recommendation(
    N: float, P: float, K: float,
    temperature: float, humidity: float,
    ph: float, rainfall: float
) -> Dict[str, Any]:
    """
    Rule-based crop recommendation (fallback for ML model).
    Uses simple heuristics based on agricultural knowledge.
    """
    logger.info("Using rule-based crop recommendation")
    
    # Simple rules (expand based on domain knowledge)
    crop = "wheat"  # default
    confidence = 0.6
    
    # Rice grows well in wet conditions
    if rainfall > 200 and humidity > 80:
        crop = "rice"
        confidence = 0.75
    
    # Cotton prefers warm, dry conditions
    elif temperature > 25 and humidity < 60:
        crop = "cotton"
        confidence = 0.7
    
    # Wheat in moderate conditions
    elif 15 < temperature < 25:
        crop = "wheat"
        confidence = 0.7
    
    # Maize in balanced conditions
    elif 20 < temperature < 30 and 60 < humidity < 80:
        crop = "maize"
        confidence = 0.7
    
    return {
        "crop": crop,
        "confidence": confidence,
        "method": "rule_based",
        "warning": "ML model unavailable, using rule-based fallback"
    }


async def rule_based_irrigation_recommendation(
    soil_moisture: float,
    temperature: float,
    humidity: float,
    crop_type: str
) -> Dict[str, Any]:
    """
    Rule-based irrigation recommendation (fallback for ML model).
    """
    logger.info("Using rule-based irrigation recommendation")
    
    # Critical thresholds
    if soil_moisture < 20:
        action = "irrigate_immediately"
        duration_minutes = 30
        urgency = "critical"
    
    elif soil_moisture < 35:
        action = "irrigate_soon"
        duration_minutes = 20
        urgency = "high"
    
    elif soil_moisture < 50:
        action = "schedule_irrigation"
        duration_minutes = 15
        urgency = "medium"
    
    else:
        action = "no_irrigation"
        duration_minutes = 0
        urgency = "low"
    
    # Adjust for temperature (hot weather needs more water)
    if temperature > 35 and action != "no_irrigation":
        duration_minutes = int(duration_minutes * 1.2)
    
    return {
        "action": action,
        "duration_minutes": duration_minutes,
        "urgency": urgency,
        "method": "rule_based",
        "warning": "ML model unavailable, using rule-based fallback"
    }


async def rule_based_pest_risk(
    temperature: float,
    humidity: float,
    rainfall: float
) -> Dict[str, Any]:
    """
    Rule-based pest risk assessment (fallback for ML model).
    """
    logger.info("Using rule-based pest risk assessment")
    
    risk_score = 0.0
    
    # High humidity increases pest risk
    if humidity > 80:
        risk_score += 0.3
    
    # Warm temperatures favor pests
    if 25 < temperature < 35:
        risk_score += 0.3
    
    # Recent rainfall can increase disease risk
    if rainfall > 50:
        risk_score += 0.2
    
    if risk_score > 0.7:
        risk_level = "high"
    elif risk_score > 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "risk_level": risk_level,
        "risk_score": min(risk_score, 1.0),
        "method": "rule_based",
        "warning": "ML model unavailable, using rule-based fallback"
    }
