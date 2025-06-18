import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.utils.redis_cache import RedisCache, redis_cache_decorator

# Add your tests for redis_cache here
def test_placeholder():
    assert True

@pytest.mark.asyncio
async def test_redis_cache_set_get(monkeypatch):
    cache = RedisCache()
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value='{"a":1}')
    mock_redis.set = AsyncMock()
    monkeypatch.setattr(cache, 'connect', AsyncMock(return_value=mock_redis))
    await cache.set('k', {'a': 1})
    val = await cache.get('k')
    assert val == {'a': 1}

@pytest.mark.asyncio
async def test_redis_cache_delete(monkeypatch):
    cache = RedisCache()
    mock_redis = MagicMock()
    mock_redis.delete = AsyncMock()
    monkeypatch.setattr(cache, 'connect', AsyncMock(return_value=mock_redis))
    await cache.delete('k')
    mock_redis.delete.assert_awaited()

@pytest.mark.asyncio
async def test_redis_cache_decorator(monkeypatch):
    cache = RedisCache()
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    monkeypatch.setattr(cache, 'connect', AsyncMock(return_value=mock_redis))
    @redis_cache_decorator(expire=1)
    async def f(x):
        return x + 1
    result = await f(1)
    assert result == 2
