import aioredis
import asyncio
import json
from typing import Any, Optional

class RedisCache:
    def __init__(self, url: str = "redis://localhost:6379/0"):
        self.url = url
        self._redis = None

    async def connect(self):
        if self._redis is None:
            self._redis = await aioredis.from_url(self.url, encoding="utf-8", decode_responses=True)
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        redis = await self.connect()
        value = await redis.get(key)
        if value is not None:
            try:
                return json.loads(value)
            except Exception:
                return value
        return None

    async def set(self, key: str, value: Any, expire: int = 300):
        redis = await self.connect()
        if not isinstance(value, str):
            value = json.dumps(value)
        await redis.set(key, value, ex=expire)

    async def delete(self, key: str):
        redis = await self.connect()
        await redis.delete(key)

# Singleton instance for DI
redis_cache = RedisCache()

# FastAPI dependency
async def get_redis_cache() -> RedisCache:
    return redis_cache
