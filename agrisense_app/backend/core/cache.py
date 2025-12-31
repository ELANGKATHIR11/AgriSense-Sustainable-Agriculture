"""
Redis Caching Layer for AgriSense
High-performance caching for sensor data, predictions, and analytics
"""
import json
import logging
from typing import Any, Optional
from functools import wraps
import hashlib

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None  # type: ignore

from ..config.optimization import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Async Redis cache manager with automatic serialization/deserialization.
    Falls back to in-memory cache if Redis is unavailable.
    """
    
    def __init__(self):
        self.redis_client: Optional[Any] = None
        self.memory_cache: dict = {}
        self.enabled = settings.enable_redis_cache and REDIS_AVAILABLE
        
    async def initialize(self):
        """Initialize Redis connection"""
        if not self.enabled:
            logger.info("Redis caching disabled, using in-memory fallback")
            return
        
        try:
            self.redis_client = await aioredis.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.enabled = False
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from prefix and arguments"""
        key_parts = [prefix] + [str(arg) for arg in args]
        if kwargs:
            key_parts.append(hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest())
        return ":".join(key_parts)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if self.enabled and self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Fallback to memory cache
        return self.memory_cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL"""
        serialized = json.dumps(value)
        
        if self.enabled and self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, serialized)
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        # Fallback to memory cache
        self.memory_cache[key] = value
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if self.enabled and self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        self.memory_cache.pop(key, None)
    
    async def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        if self.enabled and self.redis_client:
            try:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Redis clear pattern error: {e}")
        
        # For memory cache, clear matching keys
        keys_to_delete = [k for k in self.memory_cache.keys() if pattern.replace("*", "") in k]
        for key in keys_to_delete:
            del self.memory_cache[key]


# Singleton instance
cache_manager = CacheManager()


def cached(prefix: str, ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Usage:
        @cached("sensor_data", ttl=30)
        async def get_sensor_data(device_id: str):
            # expensive operation
            return data
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache_manager._generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_value = await cache_manager.get(key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {key}")
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache_ttl = ttl or settings.cache_ttl_sensor
            await cache_manager.set(key, result, cache_ttl)
            logger.debug(f"Cache miss: {key}")
            
            return result
        return wrapper
    return decorator


# Convenience functions for common cache operations

async def cache_sensor_reading(device_id: str, data: dict):
    """Cache sensor reading"""
    key = f"sensor:{device_id}:latest"
    await cache_manager.set(key, data, settings.cache_ttl_sensor)


async def get_cached_sensor_reading(device_id: str) -> Optional[dict]:
    """Get cached sensor reading"""
    key = f"sensor:{device_id}:latest"
    return await cache_manager.get(key)


async def cache_prediction(input_hash: str, prediction: dict):
    """Cache ML prediction"""
    key = f"prediction:{input_hash}"
    await cache_manager.set(key, prediction, settings.cache_ttl_prediction)


async def get_cached_prediction(input_hash: str) -> Optional[dict]:
    """Get cached ML prediction"""
    key = f"prediction:{input_hash}"
    return await cache_manager.get(key)


async def cache_analytics(query_hash: str, result: dict):
    """Cache analytics query result"""
    key = f"analytics:{query_hash}"
    await cache_manager.set(key, result, settings.cache_ttl_analytics)


async def get_cached_analytics(query_hash: str) -> Optional[dict]:
    """Get cached analytics result"""
    key = f"analytics:{query_hash}"
    return await cache_manager.get(key)
