import redis.asyncio as redis
import asyncio
import json
from typing import Any, Optional
from functools import wraps
import hashlib

class RedisCache:
    def __init__(self, url: str = "redis://localhost:6379/0"):
        self.url = url
        self._redis = None

    async def connect(self):
        if self._redis is None:
            self._redis = await redis.from_url(self.url, encoding="utf-8", decode_responses=True)
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        redis_conn = await self.connect()
        value = await redis_conn.get(key)
        if value is not None:
            try:
                return json.loads(value)
            except Exception:
                return value
        return None

    async def set(self, key: str, value: Any, expire: int = 300):
        redis_conn = await self.connect()
        if not isinstance(value, str):
            value = json.dumps(value)
        await redis_conn.set(key, value, ex=expire)

    async def delete(self, key: str):
        redis_conn = await self.connect()
        await redis_conn.delete(key)

    def redis_cache_decorator(expire: int = 300):
        """
        Decorator to cache async function results in Redis.
        The cache key is based on function name and arguments.
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create a unique cache key
                key_base = f"{func.__module__}.{func.__name__}:{args}:{kwargs}"
                cache_key = hashlib.sha256(key_base.encode()).hexdigest()
                cached = await redis_cache.get(cache_key)
                if cached is not None:
                    return cached
                result = await func(*args, **kwargs)
                await redis_cache.set(cache_key, result, expire=expire)
                return result
            return wrapper
        return decorator

# Singleton instance for DI
redis_cache = RedisCache()

# FastAPI dependency
async def get_redis_cache() -> RedisCache:
    return redis_cache
