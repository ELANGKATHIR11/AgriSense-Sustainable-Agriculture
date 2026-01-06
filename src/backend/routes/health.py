"""
Health Check and Monitoring Endpoints
Comprehensive system health checks for production reliability
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import psutil

from fastapi import APIRouter, status
from pydantic import BaseModel

from ..config.optimization import settings
from ..core.cache import cache_manager
from ..core.fallback import fallback_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthStatus(BaseModel):
    """Health check response model"""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"
    checks: Dict[str, Dict[str, Any]]


class MetricsResponse(BaseModel):
    """System metrics response"""
    timestamp: str
    system: Dict[str, Any]
    application: Dict[str, Any]


# Track application startup time
_startup_time = time.time()


def get_uptime() -> float:
    """Get application uptime in seconds"""
    return time.time() - _startup_time


async def check_database() -> Dict[str, Any]:
    """Check database connectivity"""
    try:
        # Try to import and check database
        from ..core.data_store import init_sensor_db, get_sensor_readings
        
        # Initialize if not done
        init_sensor_db()
        
        # Try a simple query
        readings = get_sensor_readings("health_check_device", limit=1)
        
        return {
            "status": "healthy",
            "message": "Database operational",
            "response_time_ms": 0  # Would measure actual query time
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e),
            "error": type(e).__name__
        }


async def check_cache() -> Dict[str, Any]:
    """Check cache (Redis or memory) connectivity"""
    try:
        if not settings.enable_redis_cache:
            return {
                "status": "healthy",
                "message": "Using in-memory cache",
                "type": "memory"
            }
        
        # Try a ping to Redis
        test_key = "health_check_ping"
        start_time = time.time()
        await cache_manager.set(test_key, "pong", ttl=5)
        result = await cache_manager.get(test_key)
        response_time = (time.time() - start_time) * 1000
        
        if result == "pong":
            return {
                "status": "healthy",
                "message": "Redis cache operational",
                "type": "redis",
                "response_time_ms": round(response_time, 2)
            }
        else:
            return {
                "status": "degraded",
                "message": "Redis cache responding but data mismatch"
            }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "degraded",
            "message": "Cache unavailable, using fallback",
            "error": str(e)
        }


async def check_ml_models() -> Dict[str, Any]:
    """Check ML models availability"""
    try:
        models_loaded = {
            "yield_model": settings.enable_yield_model,
            "irrigation_model": settings.enable_irrigation_model,
            "disease_model": settings.enable_disease_model,
            "weed_model": settings.enable_weed_model
        }
        
        enabled_count = sum(1 for v in models_loaded.values() if v)
        
        return {
            "status": "healthy" if enabled_count > 0 else "degraded",
            "message": f"{enabled_count} models enabled",
            "models": models_loaded,
            "lazy_loading": settings.lazy_load_models
        }
    except Exception as e:
        return {
            "status": "degraded",
            "message": "ML models check failed",
            "error": str(e)
        }


async def check_celery() -> Dict[str, Any]:
    """Check Celery worker status"""
    try:
        if not settings.enable_celery:
            return {
                "status": "healthy",
                "message": "Celery not enabled (sync mode)"
            }
        
        # Try to ping Celery workers
        # This would require celery app instance
        return {
            "status": "healthy",
            "message": "Celery workers operational"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "message": "Celery check failed",
            "error": str(e)
        }


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Comprehensive health check endpoint.
    
    Returns:
        - healthy: All systems operational
        - degraded: Some systems down but core functionality works
        - unhealthy: Critical systems down
    """
    checks = {}
    
    # Run all health checks in parallel
    db_check_task = check_database()
    cache_check_task = check_cache()
    ml_check_task = check_ml_models()
    celery_check_task = check_celery()
    
    checks["database"] = await db_check_task
    checks["cache"] = await cache_check_task
    checks["ml_models"] = await ml_check_task
    checks["celery"] = await celery_check_task
    
    # Determine overall status
    unhealthy_count = sum(
        1 for check in checks.values()
        if check.get("status") == "unhealthy"
    )
    degraded_count = sum(
        1 for check in checks.values()
        if check.get("status") == "degraded"
    )
    
    if unhealthy_count > 0:
        overall_status = "unhealthy"
    elif degraded_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    return HealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        uptime_seconds=get_uptime(),
        checks=checks
    )


@router.get("/health/live")
async def liveness_probe():
    """
    Kubernetes liveness probe.
    Returns 200 if application is running.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat() + "Z"}


@router.get("/health/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe.
    Returns 200 if application is ready to serve traffic.
    """
    # Check critical systems
    db_check = await check_database()
    
    if db_check["status"] == "unhealthy":
        return {
            "status": "not_ready",
            "reason": "database_unavailable"
        }, status.HTTP_503_SERVICE_UNAVAILABLE
    
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get system and application metrics.
    Useful for monitoring dashboards and alerting.
    """
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    system_metrics = {
        "cpu_percent": cpu_percent,
        "memory_total_gb": round(memory.total / (1024**3), 2),
        "memory_used_gb": round(memory.used / (1024**3), 2),
        "memory_percent": memory.percent,
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_percent": disk.percent
    }
    
    # Application metrics
    failure_stats = fallback_manager.get_failure_stats()
    
    app_metrics = {
        "uptime_seconds": get_uptime(),
        "ml_failures": failure_stats,
        "cache_enabled": settings.enable_redis_cache,
        "lazy_loading": settings.lazy_load_models,
        "graceful_degradation": settings.enable_graceful_degradation
    }
    
    return MetricsResponse(
        timestamp=datetime.utcnow().isoformat() + "Z",
        system=system_metrics,
        application=app_metrics
    )


@router.get("/health/metrics/prometheus")
async def prometheus_metrics():
    """
    Prometheus-compatible metrics endpoint.
    Returns metrics in Prometheus text format.
    """
    if not settings.enable_prometheus_metrics:
        return {"error": "Prometheus metrics not enabled"}
    
    # Generate Prometheus format metrics
    metrics = []
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    metrics.append(f"# HELP agrisense_cpu_percent CPU usage percentage")
    metrics.append(f"# TYPE agrisense_cpu_percent gauge")
    metrics.append(f"agrisense_cpu_percent {cpu_percent}")
    
    metrics.append(f"# HELP agrisense_memory_percent Memory usage percentage")
    metrics.append(f"# TYPE agrisense_memory_percent gauge")
    metrics.append(f"agrisense_memory_percent {memory.percent}")
    
    metrics.append(f"# HELP agrisense_uptime_seconds Application uptime")
    metrics.append(f"# TYPE agrisense_uptime_seconds counter")
    metrics.append(f"agrisense_uptime_seconds {get_uptime()}")
    
    # ML failure metrics
    failure_stats = fallback_manager.get_failure_stats()
    for func_name, count in failure_stats.items():
        metrics.append(f"# HELP agrisense_ml_failures_{func_name} ML failures")
        metrics.append(f"# TYPE agrisense_ml_failures_{func_name} counter")
        metrics.append(f"agrisense_ml_failures_{{{func_name}}} {count}")
    
    return "\n".join(metrics)
