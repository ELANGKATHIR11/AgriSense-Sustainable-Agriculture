"""
AgriSense Production Optimization Configuration
Centralized configuration for all optimization features
"""
import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class OptimizationSettings(BaseSettings):
    """Optimization and feature flags configuration"""
    
    # ===== ML MODEL OPTIMIZATION =====
    enable_yield_model: bool = True
    enable_irrigation_model: bool = False
    enable_disease_model: bool = True
    enable_weed_model: bool = False
    
    # Model loading strategy
    lazy_load_models: bool = True  # Load models on-demand, not at startup
    use_onnx_runtime: bool = False  # Use ONNX for inference (faster, smaller)
    use_quantized_models: bool = False  # INT8 quantization for edge devices
    
    # ===== CACHING STRATEGY =====
    enable_redis_cache: bool = False  # Set to True in production
    cache_ttl_sensor: int = 30  # seconds
    cache_ttl_prediction: int = 300  # 5 minutes
    cache_ttl_analytics: int = 600  # 10 minutes
    
    redis_url: str = "redis://localhost:6379"
    redis_max_connections: int = 10
    
    # ===== SECURITY & SAFETY =====
    # Authentication
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_exp_minutes: int = 15  # Short-lived access tokens
    jwt_refresh_exp_days: int = 7
    
    # API Security
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_ml_requests_per_minute: int = 10  # ML endpoints are expensive
    
    # Sensor validation
    enable_sensor_validation: bool = True
    sensor_min_temp: float = -20.0
    sensor_max_temp: float = 60.0
    sensor_min_humidity: float = 0.0
    sensor_max_humidity: float = 100.0
    sensor_min_moisture: float = 0.0
    sensor_max_moisture: float = 100.0
    
    # ===== PERFORMANCE =====
    # Async workers
    async_worker_pool_size: int = 4
    enable_celery: bool = False  # Set to True for production
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    
    # Database optimization
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True
    
    # ===== RELIABILITY & FAULT TOLERANCE =====
    enable_graceful_degradation: bool = True
    ml_fallback_to_rules: bool = True  # Use rule-based logic if ML fails
    cache_last_prediction: bool = True
    
    # Health checks
    health_check_interval: int = 60  # seconds
    enable_watchdog: bool = True
    
    # ===== OBSERVABILITY =====
    log_level: str = "INFO"
    enable_structured_logging: bool = True
    enable_log_sampling: bool = False
    log_sampling_rate: float = 0.1  # Sample 10% of logs
    
    # Metrics
    enable_prometheus_metrics: bool = False
    metrics_port: int = 9090
    
    # Track these metrics
    track_ml_confidence: bool = True
    track_sensor_drift: bool = True
    track_water_efficiency: bool = True
    
    # ===== COST OPTIMIZATION =====
    # Autoscaling thresholds
    autoscale_cpu_threshold: float = 70.0  # percentage
    autoscale_memory_threshold: float = 80.0
    autoscale_queue_length_threshold: int = 100
    
    # Storage tiering
    hot_storage_days: int = 30
    cold_storage_enabled: bool = False
    
    # ===== EDGE INTELLIGENCE =====
    enable_edge_mode: bool = False  # ESP32 autonomous operation
    edge_threshold_detection: bool = True
    edge_offline_buffer_size: int = 1000
    
    # ===== ALERTS & AUTOMATION =====
    enable_sms_alerts: bool = False
    enable_whatsapp_alerts: bool = False
    enable_email_alerts: bool = True
    
    # Twilio config
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_phone_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")
    
    # Alert thresholds
    alert_temperature_critical: float = 45.0
    alert_moisture_critical: float = 15.0
    alert_tank_level_low: float = 20.0
    
    # Automation
    enable_auto_irrigation: bool = False
    enable_emergency_mode: bool = True
    
    # ===== EXPLAINABLE AI =====
    enable_shap_explanations: bool = False  # Expensive, enable for premium users
    enable_lime_explanations: bool = False
    farmer_friendly_explanations: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_optimization_settings() -> OptimizationSettings:
    """
    Get cached optimization settings.
    Use lru_cache to avoid recreating settings object on every call.
    """
    return OptimizationSettings()


# Singleton instance
settings = get_optimization_settings()
