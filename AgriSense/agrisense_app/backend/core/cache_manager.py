"""
Redis Caching Strategy for AgriSense
Implements intelligent caching for sensors, ML predictions, and analytics
Part of AgriSense Production Optimization Blueprint
"""

import json
import logging
import os
import pickle
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, Optional, Union

import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# Redis configuration from environment
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# Cache TTL configurations (seconds)
CACHE_TTL_SENSOR = int(os.getenv("CACHE_TTL_SENSOR", "30"))  # 30s for sensor data
CACHE_TTL_PREDICTION = int(os.getenv("CACHE_TTL_PREDICTION", "300"))  # 5 min for ML predictions
CACHE_TTL_ANALYTICS = int(os.getenv("CACHE_TTL_ANALYTICS", "600"))  # 10 min for analytics
CACHE_TTL_CROPS = int(os.getenv("CACHE_TTL_CROPS", "3600"))  # 1 hour for crop data


class CacheManager:
    """
    Redis-based cache manager with fallback to in-memory dict
    """
    
    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        password: Optional[str] = REDIS_PASSWORD
    ):
        self.enabled = CACHE_ENABLED
        self.redis_client: Optional[redis.Redis] = None
        self._memory_cache: dict = {}  # Fallback in-memory cache
        
        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=False,  # Handle binary data
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    health_check_interval=30
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"✅ Redis connected: {host}:{port}")
            except (RedisError, Exception) as e:
                logger.warning(f"⚠️ Redis unavailable, using in-memory cache: {e}")
                self.redis_client = None
    
    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from prefix and arguments
        
        Example:
            _make_key("sensor", zone_id="Z1") -> "agrisense:sensor:zone_id:Z1"
        """
        parts = ["agrisense", prefix]
        
        # Add positional args
        for arg in args:
            parts.append(str(arg))
        
        # Add keyword args (sorted for consistency)
        for key in sorted(kwargs.keys()):
            parts.extend([key, str(kwargs[key])])
        
        return ":".join(parts)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.enabled:
            return None
        
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    # Try to unpickle (for complex objects)
                    try:
                        return pickle.loads(value)
                    except:
                        # Fallback to string decoding
                        return value.decode('utf-8')
            else:
                return self._memory_cache.get(key)
        except Exception as e:
            logger.debug(f"Cache get error for {key}: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache with optional TTL
        
        Args:
            key: Cache key
            value: Value to cache (will be pickled)
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful
        """
        if not self.enabled:
            return False
        
        try:
            # Serialize value
            if isinstance(value, (str, int, float)):
                serialized = str(value).encode('utf-8')
            else:
                serialized = pickle.dumps(value)
            
            if self.redis_client:
                if ttl:
                    self.redis_client.setex(key, ttl, serialized)
                else:
                    self.redis_client.set(key, serialized)
                return True
            else:
                # In-memory cache (no TTL support in fallback)
                self._memory_cache[key] = value
                return True
        except Exception as e:
            logger.debug(f"Cache set error for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                self._memory_cache.pop(key, None)
            return True
        except Exception as e:
            logger.debug(f"Cache delete error for {key}: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern
        
        Args:
            pattern: Pattern with wildcards (e.g., "agrisense:sensor:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
            else:
                # In-memory pattern matching
                count = 0
                pattern_prefix = pattern.replace("*", "")
                keys_to_delete = [k for k in self._memory_cache.keys() if k.startswith(pattern_prefix)]
                for key in keys_to_delete:
                    del self._memory_cache[key]
                    count += 1
                return count
        except Exception as e:
            logger.error(f"Cache delete_pattern error: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all agrisense cache entries"""
        try:
            return self.delete_pattern("agrisense:*") > 0
        except Exception as e:
            logger.error(f"Cache clear_all error: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if self.redis_client:
            try:
                info = self.redis_client.info()
                return {
                    "type": "redis",
                    "connected": True,
                    "keys": self.redis_client.dbsize(),
                    "memory_used": info.get("used_memory_human"),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0),
                }
            except Exception as e:
                return {"type": "redis", "connected": False, "error": str(e)}
        else:
            return {
                "type": "memory",
                "connected": True,
                "keys": len(self._memory_cache),
            }


# Global cache instance
_cache_manager: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(
    ttl: int = 300,
    key_prefix: str = "func",
    include_args: bool = True
):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache key
        include_args: Include function arguments in cache key
        
    Example:
        @cached(ttl=60, key_prefix="sensor_agg")
        def aggregate_sensor_data(zone_id: str, hours: int):
            # Expensive computation
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if include_args:
                cache_key = cache._make_key(key_prefix, func.__name__, *args, **kwargs)
            else:
                cache_key = cache._make_key(key_prefix, func.__name__)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"♻️ Cache hit: {cache_key}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            logger.debug(f"💾 Cached result: {cache_key}")
            
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if include_args:
                cache_key = cache._make_key(key_prefix, func.__name__, *args, **kwargs)
            else:
                cache_key = cache._make_key(key_prefix, func.__name__)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"♻️ Cache hit: {cache_key}")
                return cached_result
            
            # Execute async function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            logger.debug(f"💾 Cached result: {cache_key}")
            
            return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Predefined cache decorators for common use cases

def cache_sensor_data(ttl: int = CACHE_TTL_SENSOR):
    """Cache decorator for sensor data queries"""
    return cached(ttl=ttl, key_prefix="sensor")


def cache_ml_prediction(ttl: int = CACHE_TTL_PREDICTION):
    """Cache decorator for ML predictions"""
    return cached(ttl=ttl, key_prefix="prediction")


def cache_analytics(ttl: int = CACHE_TTL_ANALYTICS):
    """Cache decorator for dashboard analytics"""
    return cached(ttl=ttl, key_prefix="analytics")


def cache_crop_data(ttl: int = CACHE_TTL_CROPS):
    """Cache decorator for crop information"""
    return cached(ttl=ttl, key_prefix="crops")


# Example usage functions

@cache_sensor_data(ttl=30)
def get_recent_sensors_cached(zone_id: str, limit: int = 50):
    """
    Example: Cached sensor data retrieval
    Real implementation would call database
    """
    # This would call your actual data_store function
    pass


@cache_ml_prediction(ttl=300)
def predict_irrigation_cached(sensor_data: dict):
    """
    Example: Cached ML prediction
    Real implementation would call ML model
    """
    # This would call your actual ML engine
    pass


if __name__ == "__main__":
    # Test cache functionality
    cache = CacheManager()
    
    # Test basic operations
    cache.set("test:key", {"value": 123}, ttl=60)
    result = cache.get("test:key")
    print(f"Cache test result: {result}")
    
    # Test statistics
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
