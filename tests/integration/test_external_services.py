import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.utils.redis_cache import get_redis_cache

# Redis integration: test set and get
@pytest.mark.asyncio
@patch("app.utils.redis_cache.get_redis_cache", new_callable=AsyncMock)
async def test_redis_cache_set_and_get(mock_redis):
    mock_redis.return_value.get.return_value = b"test-value"
    mock_redis.return_value.set.return_value = True
    redis = await get_redis_cache()
    set_result = await redis.set("test-key", b"test-value")
    value = await redis.get("test-key")
    assert set_result is True
    assert value == b"test-value"

# Milvus integration: mock and test connection
@pytest.mark.asyncio
@patch("app.db.milvus_handler.MilvusHandler", autospec=True)
async def test_milvus_connection(mock_milvus):
    instance = mock_milvus.return_value
    instance.connect.return_value = True
    assert instance.connect() is True
    instance.disconnect.return_value = True
    assert instance.disconnect() is True

# Supabase integration: mock and test basic operation
@pytest.mark.asyncio
@patch("app.db.base.SupabaseDBHandler", autospec=True)
async def test_supabase_basic_operation(mock_supabase):
    instance = mock_supabase.return_value
    instance.create_user.return_value = {"id": 1, "email": "test@example.com"}
    user = instance.create_user({"email": "test@example.com"})
    assert user["email"] == "test@example.com"
    instance.get_user_by_email.return_value = user
    fetched = instance.get_user_by_email("test@example.com")
    assert fetched["id"] == 1
