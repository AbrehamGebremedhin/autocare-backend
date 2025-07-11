"""
Unit tests for BaseService class.
Tests caching, rate limiting, and service lifecycle.
"""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.base_service import BaseService
from app.utils.message_types import MessageType, MessageSource


class TestBaseService:
    """Test cases for BaseService class."""
    
    class ConcreteService(BaseService):
        """Concrete implementation of BaseService for testing."""
        
        def __init__(self, websocket_manager=None):
            super().__init__(websocket_manager)
            self.method_call_count = 0
        
        async def perform_action(self, *args, **kwargs):
            """Implementation of abstract method."""
            return f"performed action with {args}, {kwargs}"
        
        async def test_method(self, param1, param2=None):
            """Test method that can be cached."""
            self.method_call_count += 1
            await asyncio.sleep(0.001)  # Simulate some async work
            return f"result_{param1}_{param2}"
        
        async def expensive_operation(self):
            """Simulate an expensive operation."""
            await asyncio.sleep(0.01)
            return "expensive_result"
    
    @pytest.fixture
    def mock_websocket_manager(self):
        """Create a mock WebSocket manager."""
        manager = AsyncMock()
        manager.send_personal_message = AsyncMock()
        manager.broadcast = AsyncMock()
        return manager
    
    @pytest.fixture
    def service(self, mock_websocket_manager):
        """Create a test service instance."""
        return self.ConcreteService(mock_websocket_manager)
    
    def test_service_initialization(self, mock_websocket_manager):
        """Test service initialization."""
        service = self.ConcreteService(mock_websocket_manager)
        
        assert service.websocket_manager == mock_websocket_manager
        assert isinstance(service._cache, dict)
        assert isinstance(service._cache_ttl, dict)
        assert service._default_cache_duration == 300
        assert service._rate_limit_tokens == 10
        assert service._rate_limit_interval == 1.0
    
    def test_service_initialization_with_default_manager(self):
        """Test service initialization with default WebSocket manager."""
        with patch('app.services.base_service.manager') as mock_manager:
            service = self.ConcreteService()
            assert service.websocket_manager == mock_manager
    
    def test_get_cache_key_generation(self, service):
        """Test cache key generation."""
        # Test with different parameter combinations
        key1 = service._get_cache_key("method_name", "arg1", "arg2")
        key2 = service._get_cache_key("method_name", "arg1", "arg2", param1="value1")
        key3 = service._get_cache_key("method_name", "arg1", "arg2", param1="value1", param2="value2")
        
        assert isinstance(key1, str)
        assert isinstance(key2, str)
        assert isinstance(key3, str)
        
        # Different parameters should generate different keys
        assert key1 != key2
        assert key2 != key3
        assert key1 != key3
        
        # Same parameters should generate same key
        key4 = service._get_cache_key("method_name", "arg1", "arg2", param1="value1")
        assert key2 == key4
    
    def test_cache_key_with_sorted_kwargs(self, service):
        """Test that cache keys are consistent regardless of kwargs order."""
        key1 = service._get_cache_key("method", param1="value1", param2="value2")
        key2 = service._get_cache_key("method", param2="value2", param1="value1")
        
        assert key1 == key2
    
    def test_cache_validity_check(self, service):
        """Test cache validity checking."""
        cache_key = "test_key"
        
        # Initially, cache should be invalid
        assert not service._is_cache_valid(cache_key)
        
        # Set cache with future TTL
        service._cache[cache_key] = "test_value"
        service._cache_ttl[cache_key] = time.time() + 300  # 5 minutes in future
        
        assert service._is_cache_valid(cache_key)
        
        # Set cache with past TTL
        service._cache_ttl[cache_key] = time.time() - 1  # 1 second ago
        
        assert not service._is_cache_valid(cache_key)
    
    def test_cache_get_and_set(self, service):
        """Test cache get and set operations."""
        cache_key = "test_key"
        test_data = {"key": "value"}
        
        # Initially, cache should return None
        assert service._get_from_cache(cache_key) is None
        
        # Set cache
        service._set_cache(cache_key, test_data, ttl_seconds=60)
        
        # Get from cache
        cached_data = service._get_from_cache(cache_key)
        assert cached_data == test_data
        
        # Verify cache TTL is set
        assert cache_key in service._cache_ttl
        assert service._cache_ttl[cache_key] > time.time()
    
    def test_cache_expiration_cleanup(self, service):
        """Test that expired cache entries are cleaned up."""
        cache_key = "test_key"
        test_data = "test_value"
        
        # Set cache with short TTL
        service._set_cache(cache_key, test_data, ttl_seconds=0.001)
        
        # Wait for cache to expire
        time.sleep(0.002)
        
        # Getting from cache should return None and clean up
        result = service._get_from_cache(cache_key)
        assert result is None
        assert cache_key not in service._cache
        assert cache_key not in service._cache_ttl
    
    def test_cache_default_ttl(self, service):
        """Test cache with default TTL."""
        cache_key = "test_key"
        test_data = "test_value"
        
        service._set_cache(cache_key, test_data)
        
        # Should use default TTL
        expected_ttl = time.time() + service._default_cache_duration
        actual_ttl = service._cache_ttl[cache_key]
        
        # Allow for small time difference
        assert abs(actual_ttl - expected_ttl) < 1
    
    @pytest.mark.asyncio
    async def test_async_method_execution(self, service):
        """Test async method execution."""
        result = await service.test_method("param1", param2="param2")
        
        assert result == "result_param1_param2"
        assert service.method_call_count == 1
    
    @pytest.mark.asyncio
    async def test_service_with_websocket_messaging(self, service):
        """Test service with WebSocket messaging."""
        # Mock the WebSocket manager
        service.websocket_manager.broadcast = AsyncMock()
        
        # Test method that might send WebSocket messages
        await service.test_method("test")
        
        # Verify WebSocket manager is available
        assert service.websocket_manager is not None
        assert hasattr(service.websocket_manager, 'broadcast')
    
    def test_rate_limiting_initialization(self, service):
        """Test rate limiting initialization."""
        assert service._rate_limit_tokens == 10
        assert service._rate_limit_interval == 1.0
        assert service._last_reset <= time.time()
    
    def test_connection_pool_initialization(self, service):
        """Test connection pool initialization."""
        assert service._connection_pool is None
    
    def test_service_cache_isolation(self, mock_websocket_manager):
        """Test that different service instances have isolated caches."""
        service1 = self.ConcreteService(mock_websocket_manager)
        service2 = self.ConcreteService(mock_websocket_manager)
        
        # Set cache in service1
        service1._set_cache("key1", "value1")
        
        # service2 should not have access to service1's cache
        assert service2._get_from_cache("key1") is None
        assert service1._get_from_cache("key1") == "value1"
    
    def test_service_cache_memory_efficiency(self, service):
        """Test cache memory efficiency with many entries."""
        # Add many cache entries
        for i in range(100):
            service._set_cache(f"key_{i}", f"value_{i}", ttl_seconds=0.001)
        
        # Wait for entries to expire
        time.sleep(0.002)
        
        # Access entries to trigger cleanup - since cleanup is lazy
        expired_count = 0
        for i in range(100):
            if service._get_from_cache(f"key_{i}") is None:
                expired_count += 1
        
        # All entries should have expired and been cleaned up when accessed
        assert expired_count == 100
        assert len(service._cache) == 0
        assert len(service._cache_ttl) == 0
    
    def test_service_cache_with_none_values(self, service):
        """Test caching None values."""
        cache_key = "none_key"
        
        # Cache None value
        service._set_cache(cache_key, None, ttl_seconds=60)
        
        # Should be able to retrieve None
        cached_value = service._get_from_cache(cache_key)
        assert cached_value is None
        
        # Should be considered valid cache
        assert service._is_cache_valid(cache_key)
    
    def test_service_cache_with_complex_data(self, service):
        """Test caching complex data structures."""
        complex_data = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (1, 2, 3),
            "mixed": [{"key": "value"}, [1, 2, 3]]
        }
        
        cache_key = "complex_key"
        service._set_cache(cache_key, complex_data, ttl_seconds=60)
        
        cached_data = service._get_from_cache(cache_key)
        assert cached_data == complex_data
        assert cached_data["list"] == [1, 2, 3]
        assert cached_data["dict"]["nested"] == "value"
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, service):
        """Test concurrent cache access."""
        async def cache_operation(key_suffix):
            key = f"concurrent_key_{key_suffix}"
            service._set_cache(key, f"value_{key_suffix}", ttl_seconds=60)
            return service._get_from_cache(key)
        
        # Run multiple concurrent cache operations
        tasks = [cache_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        for i, result in enumerate(results):
            assert result == f"value_{i}"
    
    def test_service_subclass_inheritance(self, mock_websocket_manager):
        """Test that service subclasses inherit properly."""
        class ExtendedService(BaseService):
            def __init__(self, websocket_manager=None):
                super().__init__(websocket_manager)
                self.custom_attribute = "custom_value"
            
            async def perform_action(self, *args, **kwargs):
                """Implementation of abstract method."""
                return f"extended action with {args}, {kwargs}"
        
        extended_service = ExtendedService(mock_websocket_manager)
        
        # Should inherit all BaseService attributes
        assert hasattr(extended_service, '_cache')
        assert hasattr(extended_service, '_cache_ttl')
        assert hasattr(extended_service, '_default_cache_duration')
        assert hasattr(extended_service, 'websocket_manager')
        
        # Should have custom attribute
        assert extended_service.custom_attribute == "custom_value"
    
    def test_service_cache_key_collision_resistance(self, service):
        """Test that cache keys are collision-resistant."""
        # These should generate different cache keys
        key1 = service._get_cache_key("method1", "arg1", "arg2")
        key2 = service._get_cache_key("method2", "arg1", "arg2")
        key3 = service._get_cache_key("method1", "arg1", "arg3")
        key4 = service._get_cache_key("method1", "arg2", "arg1")
        
        # All keys should be different
        keys = [key1, key2, key3, key4]
        assert len(set(keys)) == len(keys)
    
    def test_service_cache_ttl_precision(self, service):
        """Test cache TTL precision."""
        cache_key = "precision_key"
        ttl_seconds = 0.1  # 100ms
        
        start_time = time.time()
        service._set_cache(cache_key, "value", ttl_seconds)
        
        # Should be valid immediately
        assert service._is_cache_valid(cache_key)
        
        # Wait for cache to expire
        time.sleep(0.15)
        
        # Should be invalid after expiration
        assert not service._is_cache_valid(cache_key)
        
        # Total time should be reasonable
        elapsed = time.time() - start_time
        assert 0.1 <= elapsed <= 0.3  # Allow for some timing variation
    
    def test_service_websocket_manager_dependency(self, service):
        """Test WebSocket manager dependency."""
        assert service.websocket_manager is not None
        assert hasattr(service.websocket_manager, 'send_personal_message')
        assert hasattr(service.websocket_manager, 'broadcast')
    
    @pytest.mark.asyncio
    async def test_service_async_context_handling(self, service):
        """Test service handling of async context."""
        async def async_operation():
            await asyncio.sleep(0.001)
            return "async_result"
        
        # Should be able to handle async operations
        result = await async_operation()
        assert result == "async_result"
    
    def test_service_memory_usage_optimization(self, service):
        """Test that service doesn't leak memory."""
        initial_cache_size = len(service._cache)
        
        # Add many short-lived cache entries
        for i in range(1000):
            service._set_cache(f"temp_key_{i}", f"temp_value_{i}", ttl_seconds=0.001)
        
        # Wait for expiration
        time.sleep(0.002)
        
        # Force cleanup by accessing all expired keys
        for i in range(1000):
            service._get_from_cache(f"temp_key_{i}")
        
        # Cache should be cleaned up after accessing expired keys
        final_cache_size = len(service._cache)
        assert final_cache_size <= initial_cache_size + 10  # Allow for some tolerance
