import redis.asyncio as redis
import asyncio
import json
from typing import Any, Optional
from functools import wraps
import hashlib
from app.core.config import Settings
from app.core.interfaces import Protocol
from app.utils.exceptions import ExternalServiceException
from app.utils.logger import get_logger_instance
from contextlib import asynccontextmanager
import time

class IRedisCache(Protocol):
    async def connect(self): ...
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, expire: int = 300): ...
    async def delete(self, key: str): ...
    async def lpush(self, key: str, *values): ...
    async def expire(self, key: str, seconds: int): ...
    async def health_check(self) -> bool: ...
    async def close(self): ...

class RedisCache(IRedisCache):
    def __init__(self, url: str = None):
        self.logger = get_logger_instance("RedisCache")
        if url is None:
            settings = Settings()
            host = settings.REDIS_HOST
            port = settings.REDIS_PORT
            url = f"redis://{host}:{port}/0"
        self.url = url
        self._pool = None
        self._connection_pool_size = 10
        self._connection_timeout = 5
        self._retry_attempts = 3
        self._retry_delay = 1

    async def _create_pool(self):
        """Create Redis connection pool"""
        try:
            self._pool = redis.ConnectionPool.from_url(
                self.url,
                max_connections=self._connection_pool_size,
                retry_on_timeout=True,
                socket_timeout=self._connection_timeout,
                socket_connect_timeout=self._connection_timeout,
                encoding="utf-8",
                decode_responses=True
            )
            await self.logger.info("Redis connection pool created successfully")
            return self._pool
        except Exception as e:
            await self.logger.error(f"Failed to create Redis connection pool: {str(e)}")
            raise ExternalServiceException("Redis", f"Connection pool creation failed: {str(e)}")

    async def connect(self):
        """Get Redis connection from pool"""
        if self._pool is None:
            await self._create_pool()
        
        for attempt in range(self._retry_attempts):
            try:
                connection = redis.Redis(connection_pool=self._pool)
                # Test the connection
                await connection.ping()
                return connection
            except Exception as e:
                await self.logger.warning(f"Redis connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == self._retry_attempts - 1:
                    raise ExternalServiceException("Redis", f"Connection failed after {self._retry_attempts} attempts")
                await asyncio.sleep(self._retry_delay * (attempt + 1))

    @asynccontextmanager
    async def get_connection(self):
        """Context manager for Redis connections"""
        connection = None
        try:
            connection = await self.connect()
            yield connection
        except Exception as e:
            await self.logger.error(f"Redis operation failed: {str(e)}")
            raise ExternalServiceException("Redis", str(e))
        finally:
            if connection:
                try:
                    await connection.close()
                except Exception as e:
                    await self.logger.warning(f"Error closing Redis connection: {str(e)}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis with error handling"""
        try:
            async with self.get_connection() as redis_conn:
                value = await redis_conn.get(key)
                if value is not None:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value
                return None
        except ExternalServiceException:
            raise
        except Exception as e:
            await self.logger.error(f"Redis GET operation failed for key '{key}': {str(e)}")
            # Return None for cache misses due to errors (graceful degradation)
            return None

    async def set(self, key: str, value: Any, expire: int = 300):
        """Set value in Redis with error handling"""
        try:
            async with self.get_connection() as redis_conn:
                # Handle AIMessage objects from LangChain
                if hasattr(value, "content") and hasattr(value, "type"):
                    # This is likely a LangChain message object
                    value = {"content": value.content, "type": value.type}
                
                # Ensure value is JSON serializable
                if not isinstance(value, str):
                    try:
                        value = json.dumps(value)
                    except TypeError as e:
                        # If JSON serialization fails, convert to string
                        await self.logger.error(f"JSON serialization failed: {str(e)} - converting to string")
                        value = str(value)
                
                await redis_conn.set(key, value, ex=expire)
        except ExternalServiceException:
            raise
        except Exception as e:
            await self.logger.error(f"Redis SET operation failed for key '{key}': {str(e)}")
            # Don't raise exception for cache write failures (graceful degradation)

    async def delete(self, key: str):
        """Delete value from Redis with error handling"""
        try:
            async with self.get_connection() as redis_conn:
                await redis_conn.delete(key)
        except ExternalServiceException:
            raise
        except Exception as e:
            await self.logger.error(f"Redis DELETE operation failed for key '{key}': {str(e)}")

    async def lpush(self, key: str, *values):
        """Push values to the head of a list in Redis"""
        try:
            async with self.get_connection() as redis_conn:
                await redis_conn.lpush(key, *values)
        except ExternalServiceException:
            raise
        except Exception as e:
            await self.logger.error(f"Redis LPUSH operation failed for key '{key}': {str(e)}")

    async def expire(self, key: str, seconds: int):
        """Set expiration time for a key in Redis"""
        try:
            async with self.get_connection() as redis_conn:
                await redis_conn.expire(key, seconds)
        except ExternalServiceException:
            raise
        except Exception as e:
            await self.logger.error(f"Redis EXPIRE operation failed for key '{key}': {str(e)}")

    async def health_check(self) -> bool:
        """Perform Redis health check"""
        try:
            async with self.get_connection() as redis_conn:
                await redis_conn.ping()
                return True
        except Exception as e:
            await self.logger.error(f"Redis health check failed: {str(e)}")
            return False

    async def close(self):
        """Close Redis connection pool"""
        try:
            if self._pool:
                await self._pool.disconnect()
                self._pool = None
                await self.logger.info("Redis connection pool closed")
        except Exception as e:
            await self.logger.error(f"Error closing Redis connection pool: {str(e)}")

    async def get_stats(self) -> dict:
        """Get Redis connection statistics"""
        try:
            async with self.get_connection() as redis_conn:
                info = await redis_conn.info()
                return {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                }
        except Exception as e:
            await self.logger.error(f"Failed to get Redis stats: {str(e)}")
            return {}

# Singleton instance for DI
redis_cache = RedisCache()

def redis_cache_decorator(expire: int = 300, key_prefix: str = ""):
    """
    Enhanced decorator to cache async function results in Redis with error handling
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a unique cache key
            key_base = f"{key_prefix}:{func.__module__}.{func.__name__}:{args}:{kwargs}"
            cache_key = hashlib.sha256(key_base.encode()).hexdigest()[:32]  # Limit key length
            
            try:
                cached = await redis_cache.get(cache_key)
                if cached is not None:
                    return cached
            except Exception as e:
                # Cache read error - continue without cache
                pass
            
            # Execute function
            result = await func(*args, **kwargs)
            
            try:
                await redis_cache.set(cache_key, result, expire=expire)
            except Exception as e:
                # Cache write error - continue without caching
                pass
            
            return result
        return wrapper
    return decorator

async def get_redis_cache() -> RedisCache:
    return redis_cache

__all__ = ["RedisCache", "redis_cache", "redis_cache_decorator", "get_redis_cache"]
