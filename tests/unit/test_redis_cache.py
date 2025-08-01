"""
Unit tests for Redis cache implementation and connection management.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import json
import hashlib

from app.utils.redis_cache import (
    RedisCache, 
    redis_cache, 
    redis_cache_decorator, 
    get_redis_cache
)
from app.utils.exceptions import ExternalServiceException


class TestRedisCache:
    """Test Redis cache implementation"""
    
    @pytest.fixture
    def redis_cache_instance(self):
        """Create Redis cache instance"""
        return RedisCache("redis://localhost:6379/0")
    
    @pytest.fixture
    def mock_redis_pool(self):
        """Mock Redis connection pool"""
        pool = MagicMock()
        return pool
    
    @pytest.fixture
    def mock_redis_connection(self):
        """Mock Redis connection"""
        connection = MagicMock()
        connection.ping = AsyncMock(return_value=True)
        connection.get = AsyncMock()
        connection.set = AsyncMock()
        connection.delete = AsyncMock()
        connection.close = AsyncMock()
        connection.info = AsyncMock(return_value={
            "connected_clients": 5,
            "used_memory": 1024000,
            "used_memory_human": "1.02M",
            "keyspace_hits": 100,
            "keyspace_misses": 20
        })
        return connection
    
    def test_initialization_default_url(self):
        """Test initialization with default URL from settings"""
        with patch('app.utils.redis_cache.Settings') as mock_settings_class:
            mock_settings = MagicMock()
            mock_settings.REDIS_HOST = "test-host"
            mock_settings.REDIS_PORT = "1234"
            mock_settings_class.return_value = mock_settings
            
            cache = RedisCache()
            assert cache.url == "redis://test-host:1234/0"
    
    def test_initialization_custom_url(self):
        """Test initialization with custom URL"""
        custom_url = "redis://custom-host:5678/1"
        cache = RedisCache(custom_url)
        assert cache.url == custom_url
    
    @patch('app.utils.redis_cache.redis.ConnectionPool')
    def test_create_pool_success(self, mock_pool_class, redis_cache_instance):
        """Test successful connection pool creation"""
        mock_pool = MagicMock()
        mock_pool_class.from_url.return_value = mock_pool
        
        pool = asyncio.run(redis_cache_instance._create_pool())
        
        assert pool == mock_pool
        mock_pool_class.from_url.assert_called_once_with(
            redis_cache_instance.url,
            max_connections=redis_cache_instance._connection_pool_size,
            retry_on_timeout=True,
            socket_timeout=redis_cache_instance._connection_timeout,
            socket_connect_timeout=redis_cache_instance._connection_timeout,
            encoding="utf-8",
            decode_responses=True
        )
    
    @patch('app.utils.redis_cache.redis.ConnectionPool')
    def test_create_pool_failure(self, mock_pool_class, redis_cache_instance):
        """Test connection pool creation failure"""
        mock_pool_class.from_url.side_effect = Exception("Connection failed")
        
        with pytest.raises(ExternalServiceException) as exc_info:
            asyncio.run(redis_cache_instance._create_pool())
        
        assert "Redis" in str(exc_info.value.detail)
        assert "Connection pool creation failed" in str(exc_info.value.detail)
    
    @patch('app.utils.redis_cache.redis.Redis')
    def test_connect_success(self, mock_redis_class, redis_cache_instance, mock_redis_connection):
        """Test successful Redis connection"""
        mock_redis_class.return_value = mock_redis_connection
        
        with patch.object(redis_cache_instance, '_create_pool', new_callable=AsyncMock) as mock_create_pool:
            mock_pool = MagicMock()
            mock_create_pool.return_value = mock_pool
            
            connection = asyncio.run(redis_cache_instance.connect())
            
            assert connection == mock_redis_connection
            mock_redis_class.assert_called_once_with(connection_pool=mock_pool)
            mock_redis_connection.ping.assert_called_once()
    
    def test_connect_with_retry(self, redis_cache_instance):
        """Test connection with retry logic"""
        with patch('app.utils.redis_cache.redis.Redis') as mock_redis_class:
            mock_connection = MagicMock()
            
            # First attempt fails, second succeeds
            mock_connection.ping = AsyncMock(side_effect=[Exception("Connection failed"), True])
            mock_redis_class.return_value = mock_connection
            
            with patch.object(redis_cache_instance, '_create_pool', new_callable=AsyncMock):
                connection = asyncio.run(redis_cache_instance.connect())
                
                assert connection == mock_connection
                assert mock_connection.ping.call_count == 2
    
    def test_connect_max_retries_exceeded(self, redis_cache_instance):
        """Test connection when max retries exceeded"""
        with patch('app.utils.redis_cache.redis.Redis') as mock_redis_class:
            mock_connection = MagicMock()
            mock_connection.ping = AsyncMock(side_effect=Exception("Connection failed"))
            mock_redis_class.return_value = mock_connection
            
            with patch.object(redis_cache_instance, '_create_pool', new_callable=AsyncMock):
                with pytest.raises(ExternalServiceException) as exc_info:
                    asyncio.run(redis_cache_instance.connect())
                
                assert f"Connection failed after {redis_cache_instance._retry_attempts} attempts" in str(exc_info.value.detail)
    
    def test_get_connection_context_manager(self, redis_cache_instance, mock_redis_connection):
        """Test connection context manager"""
        with patch.object(redis_cache_instance, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_redis_connection
            
            async def test_context():
                async with redis_cache_instance.get_connection() as conn:
                    assert conn == mock_redis_connection
                
                # Connection should be closed after context
                mock_redis_connection.close.assert_called_once()
            
            asyncio.run(test_context())
    
    def test_get_connection_error_handling(self, redis_cache_instance):
        """Test connection context manager error handling"""
        with patch.object(redis_cache_instance, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            async def test_error():
                with pytest.raises(ExternalServiceException):
                    async with redis_cache_instance.get_connection():
                        pass
            
            asyncio.run(test_error())
    
    def test_get_success(self, redis_cache_instance, mock_redis_connection):
        """Test successful get operation"""
        test_data = {"key": "value", "number": 123}
        mock_redis_connection.get.return_value = json.dumps(test_data)
        
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_redis_connection)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = asyncio.run(redis_cache_instance.get("test_key"))
            
            assert result == test_data
            mock_redis_connection.get.assert_called_once_with("test_key")
    
    def test_get_string_value(self, redis_cache_instance, mock_redis_connection):
        """Test get operation with string value"""
        mock_redis_connection.get.return_value = "simple_string"
        
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_redis_connection)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = asyncio.run(redis_cache_instance.get("test_key"))
            
            assert result == "simple_string"
    
    def test_get_not_found(self, redis_cache_instance, mock_redis_connection):
        """Test get operation when key not found"""
        mock_redis_connection.get.return_value = None
        
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_redis_connection)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = asyncio.run(redis_cache_instance.get("nonexistent_key"))
            
            assert result is None
    
    def test_get_error_graceful_degradation(self, redis_cache_instance):
        """Test get operation graceful degradation on error"""
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.side_effect = Exception("Redis error")
            
            result = asyncio.run(redis_cache_instance.get("test_key"))
            
            # Should return None instead of raising exception
            assert result is None
    
    def test_set_success(self, redis_cache_instance, mock_redis_connection):
        """Test successful set operation"""
        test_data = {"key": "value"}
        
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_redis_connection)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=None)
            
            asyncio.run(redis_cache_instance.set("test_key", test_data, expire=300))
            
            mock_redis_connection.set.assert_called_once_with(
                "test_key", 
                json.dumps(test_data), 
                ex=300
            )
    
    def test_set_string_value(self, redis_cache_instance, mock_redis_connection):
        """Test set operation with string value"""
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_redis_connection)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=None)
            
            asyncio.run(redis_cache_instance.set("test_key", "string_value"))
            
            mock_redis_connection.set.assert_called_once_with(
                "test_key", 
                "string_value", 
                ex=300  # default expire
            )
    
    def test_set_error_graceful_degradation(self, redis_cache_instance):
        """Test set operation graceful degradation on error"""
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.side_effect = Exception("Redis error")
            
            # Should not raise exception
            asyncio.run(redis_cache_instance.set("test_key", "value"))
    
    def test_delete_success(self, redis_cache_instance, mock_redis_connection):
        """Test successful delete operation"""
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_redis_connection)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=None)
            
            asyncio.run(redis_cache_instance.delete("test_key"))
            
            mock_redis_connection.delete.assert_called_once_with("test_key")
    
    def test_health_check_success(self, redis_cache_instance, mock_redis_connection):
        """Test successful health check"""
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_redis_connection)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = asyncio.run(redis_cache_instance.health_check())
            
            assert result == True
            mock_redis_connection.ping.assert_called_once()
    
    def test_health_check_failure(self, redis_cache_instance):
        """Test health check failure"""
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.side_effect = Exception("Connection failed")
            
            result = asyncio.run(redis_cache_instance.health_check())
            
            assert result == False
    
    def test_close_success(self, redis_cache_instance):
        """Test successful close operation"""
        mock_pool = MagicMock()
        mock_pool.disconnect = AsyncMock()
        redis_cache_instance._pool = mock_pool
        
        asyncio.run(redis_cache_instance.close())
        
        mock_pool.disconnect.assert_called_once()
        assert redis_cache_instance._pool is None
    
    def test_close_error_handling(self, redis_cache_instance):
        """Test close operation error handling"""
        mock_pool = MagicMock()
        mock_pool.disconnect = AsyncMock(side_effect=Exception("Close error"))
        redis_cache_instance._pool = mock_pool
        
        # Should not raise exception
        asyncio.run(redis_cache_instance.close())
    
    def test_get_stats_success(self, redis_cache_instance, mock_redis_connection):
        """Test successful stats retrieval"""
        expected_stats = {
            "connected_clients": 5,
            "used_memory": 1024000,
            "used_memory_human": "1.02M",
            "keyspace_hits": 100,
            "keyspace_misses": 20
        }
        
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_redis_connection)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=None)
            
            stats = asyncio.run(redis_cache_instance.get_stats())
            
            assert stats == expected_stats
            mock_redis_connection.info.assert_called_once()
    
    def test_get_stats_failure(self, redis_cache_instance):
        """Test stats retrieval failure"""
        with patch.object(redis_cache_instance, 'get_connection') as mock_get_conn:
            mock_get_conn.side_effect = Exception("Stats error")
            
            stats = asyncio.run(redis_cache_instance.get_stats())
            
            assert stats == {}


class TestRedisCacheDecorator:
    """Test Redis cache decorator"""
    
    @pytest.fixture
    def mock_redis_cache(self):
        """Mock Redis cache for decorator tests"""
        cache = MagicMock()
        cache.get = AsyncMock()
        cache.set = AsyncMock()
        return cache
    
    def test_decorator_cache_hit(self, mock_redis_cache):
        """Test decorator with cache hit"""
        cached_result = {"cached": True}
        mock_redis_cache.get.return_value = cached_result
        
        @redis_cache_decorator(expire=600, key_prefix="test")
        async def test_function(arg1, arg2, kwarg1=None):
            return {"fresh": True}
        
        with patch('app.utils.redis_cache.redis_cache', mock_redis_cache):
            result = asyncio.run(test_function("value1", "value2", kwarg1="kwvalue"))
            
            assert result == cached_result
            mock_redis_cache.get.assert_called_once()
            mock_redis_cache.set.assert_not_called()
    
    def test_decorator_cache_miss(self, mock_redis_cache):
        """Test decorator with cache miss"""
        mock_redis_cache.get.return_value = None
        fresh_result = {"fresh": True}
        
        @redis_cache_decorator(expire=600, key_prefix="test")
        async def test_function(arg1, arg2):
            return fresh_result
        
        with patch('app.utils.redis_cache.redis_cache', mock_redis_cache):
            result = asyncio.run(test_function("value1", "value2"))
            
            assert result == fresh_result
            mock_redis_cache.get.assert_called_once()
            mock_redis_cache.set.assert_called_once()
    
    def test_decorator_key_generation(self, mock_redis_cache):
        """Test decorator cache key generation"""
        mock_redis_cache.get.return_value = None
        
        @redis_cache_decorator(expire=300, key_prefix="test_prefix")
        async def test_function(arg1, arg2, kwarg1=None):
            return {"result": True}
        
        with patch('app.utils.redis_cache.redis_cache', mock_redis_cache):
            asyncio.run(test_function("arg_value", 123, kwarg1="kw_value"))
            
            # Check that get was called with a hashed key
            call_args = mock_redis_cache.get.call_args[0]
            cache_key = call_args[0]
            
            assert isinstance(cache_key, str)
            assert len(cache_key) == 32  # SHA256 hash truncated to 32 chars
    
    def test_decorator_cache_error_handling(self, mock_redis_cache):
        """Test decorator error handling"""
        mock_redis_cache.get.side_effect = Exception("Cache error")
        mock_redis_cache.set.side_effect = Exception("Cache error")
        
        fresh_result = {"fresh": True}
        
        @redis_cache_decorator(expire=300)
        async def test_function():
            return fresh_result
        
        with patch('app.utils.redis_cache.redis_cache', mock_redis_cache):
            result = asyncio.run(test_function())
            
            # Should return fresh result despite cache errors
            assert result == fresh_result
    
    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata"""
        @redis_cache_decorator(expire=300)
        async def test_function(arg1, arg2):
            """Test function docstring"""
            return {"result": True}
        
        assert test_function.__name__ == "test_function"
        # Note: functools.wraps should preserve docstring, but we're not testing that here
        # as it depends on the implementation details


class TestGlobalRedisCacheInstance:
    """Test global Redis cache instance"""
    
    def test_global_instance_exists(self):
        """Test that global redis_cache instance exists"""
        from app.utils.redis_cache import redis_cache as global_cache
        assert global_cache is not None
        assert isinstance(global_cache, RedisCache)
    
    def test_get_redis_cache_dependency(self):
        """Test get_redis_cache dependency function"""
        from app.utils.redis_cache import redis_cache as global_cache
        
        result = asyncio.run(get_redis_cache())
        assert result is global_cache


class TestRedisCacheIntegration:
    """Integration tests for Redis cache"""
    
    def test_full_cache_cycle(self):
        """Test complete cache cycle: set, get, delete"""
        cache = RedisCache("redis://localhost:6379/0")
        test_data = {"integration": "test", "number": 42}
        
        with patch.object(cache, 'get_connection') as mock_get_conn:
            mock_redis = MagicMock()
            mock_redis.get = AsyncMock(side_effect=[None, json.dumps(test_data), None])
            mock_redis.set = AsyncMock()
            mock_redis.delete = AsyncMock()
            
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_redis)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=None)
            
            async def test_cycle():
                # Initially not in cache
                result = await cache.get("integration_key")
                assert result is None
                
                # Set in cache
                await cache.set("integration_key", test_data, expire=300)
                
                # Should now be in cache
                result = await cache.get("integration_key")
                assert result == test_data
                
                # Delete from cache
                await cache.delete("integration_key")
                
                # Should no longer be in cache
                result = await cache.get("integration_key")
                assert result is None
            
            asyncio.run(test_cycle())
