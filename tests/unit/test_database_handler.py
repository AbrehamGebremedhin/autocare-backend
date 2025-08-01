"""
Unit tests for database handler and connection management.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import time

from app.db.base import SupabaseDBHandler, get_db_handler
from app.utils.exceptions import DatabaseException, ConfigurationException


class TestSupabaseDBHandler:
    """Test Supabase database handler"""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings with valid configuration"""
        settings = MagicMock()
        settings.SUPABASE_URL = "https://test.supabase.co"
        settings.SUPABASE_KEY = "test-key-123"
        return settings
    
    @pytest.fixture
    def invalid_settings(self):
        """Mock settings with invalid configuration"""
        settings = MagicMock()
        settings.SUPABASE_URL = ""
        settings.SUPABASE_KEY = ""
        return settings
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton instance before each test"""
        SupabaseDBHandler._instance = None
        SupabaseDBHandler._client = None
        SupabaseDBHandler._connection_pool = {}
        yield
        SupabaseDBHandler._instance = None
        SupabaseDBHandler._client = None
        SupabaseDBHandler._connection_pool = {}
    
    @patch('app.db.base.get_settings')
    def test_singleton_pattern(self, mock_get_settings, mock_settings):
        """Test that handler follows singleton pattern"""
        mock_get_settings.return_value = mock_settings
        
        handler1 = SupabaseDBHandler()
        handler2 = SupabaseDBHandler()
        
        assert handler1 is handler2
        assert SupabaseDBHandler._instance is handler1
    
    @patch('app.db.base.get_settings')
    def test_initialization_valid_config(self, mock_get_settings, mock_settings):
        """Test initialization with valid configuration"""
        mock_get_settings.return_value = mock_settings
        
        handler = SupabaseDBHandler()
        
        assert handler.settings == mock_settings
        assert hasattr(handler, 'logger')
    
    @patch('app.db.base.get_settings')
    def test_initialization_invalid_config(self, mock_get_settings, invalid_settings):
        """Test initialization with invalid configuration"""
        mock_get_settings.return_value = invalid_settings
        
        with pytest.raises(ConfigurationException) as exc_info:
            SupabaseDBHandler()
        
        assert "Supabase URL and Key must be set" in str(exc_info.value.detail)
        assert exc_info.value.detail["missing_vars"] == ["SUPABASE_URL", "SUPABASE_KEY"]
    
    @patch('app.db.base.get_settings')
    @patch('app.db.base.create_client')
    def test_create_client_success(self, mock_create_client, mock_get_settings, mock_settings):
        """Test successful client creation"""
        mock_get_settings.return_value = mock_settings
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        handler = SupabaseDBHandler()
        
        with patch.object(handler, '_test_connection', new_callable=AsyncMock) as mock_test:
            mock_test.return_value = None
            
            client = asyncio.run(handler._create_client())
            
            assert client == mock_client
            mock_create_client.assert_called_once_with(
                mock_settings.SUPABASE_URL, 
                mock_settings.SUPABASE_KEY
            )
            mock_test.assert_called_once_with(mock_client)
    
    @patch('app.db.base.get_settings')
    @patch('app.db.base.create_client')
    def test_create_client_with_retry(self, mock_create_client, mock_get_settings, mock_settings):
        """Test client creation with retry logic"""
        mock_get_settings.return_value = mock_settings
        mock_client = MagicMock()
        
        # First call fails, second succeeds
        mock_create_client.side_effect = [Exception("Connection failed"), mock_client]
        
        handler = SupabaseDBHandler()
        
        with patch.object(handler, '_test_connection', new_callable=AsyncMock) as mock_test:
            mock_test.return_value = None
            
            client = asyncio.run(handler._create_client())
            
            assert client == mock_client
            assert mock_create_client.call_count == 2
    
    @patch('app.db.base.get_settings')
    @patch('app.db.base.create_client')
    def test_create_client_max_retries_exceeded(self, mock_create_client, mock_get_settings, mock_settings):
        """Test client creation when max retries exceeded"""
        mock_get_settings.return_value = mock_settings
        mock_create_client.side_effect = Exception("Connection failed")
        
        handler = SupabaseDBHandler()
        
        with pytest.raises(DatabaseException) as exc_info:
            asyncio.run(handler._create_client())
        
        assert "Failed to establish database connection" in str(exc_info.value.detail)
        assert exc_info.value.detail["attempts"] == handler._retry_attempts
        assert mock_create_client.call_count == handler._retry_attempts
    
    @patch('app.db.base.get_settings')
    def test_test_connection(self, mock_get_settings, mock_settings):
        """Test connection testing"""
        mock_get_settings.return_value = mock_settings
        handler = SupabaseDBHandler()
        
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.limit.return_value = mock_table
        mock_table.execute.return_value = MagicMock()
        
        # Should not raise exception
        asyncio.run(handler._test_connection(mock_client))
        
        mock_client.table.assert_called_once_with("ping")
    
    @patch('app.db.base.get_settings')
    @patch('app.db.base.create_client')
    def test_get_connection_context_manager(self, mock_create_client, mock_get_settings, mock_settings):
        """Test connection context manager"""
        mock_get_settings.return_value = mock_settings
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        handler = SupabaseDBHandler()
        
        with patch.object(handler, '_test_connection', new_callable=AsyncMock):
            async def test_context_manager():
                # Pool should be empty initially
                assert len(handler._connection_pool) == 0
                
                async with handler.get_connection() as client:
                    assert client == mock_client
                    # Inside the context, connection should be in pool
                    assert len(handler._connection_pool) == 1
                
                # After context, connection should be removed
                assert len(handler._connection_pool) == 0
            
            asyncio.run(test_context_manager())
    
    @patch('app.db.base.get_settings')
    def test_get_connection_error_handling(self, mock_get_settings, mock_settings):
        """Test connection error handling"""
        mock_get_settings.return_value = mock_settings
        handler = SupabaseDBHandler()
        
        with patch.object(handler, '_create_client', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Database error")
            
            async def test_error_handling():
                with pytest.raises(DatabaseException) as exc_info:
                    async with handler.get_connection():
                        pass
                
                assert "Database connection failed" in str(exc_info.value.detail)
            
            asyncio.run(test_error_handling())
    
    @patch('app.db.base.get_settings')
    @patch('app.db.base.create_client')
    def test_client_property_backward_compatibility(self, mock_create_client, mock_get_settings, mock_settings):
        """Test client property for backward compatibility"""
        mock_get_settings.return_value = mock_settings
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        handler = SupabaseDBHandler()
        
        with patch.object(handler, '_test_connection', new_callable=AsyncMock):
            client = asyncio.run(handler.client)
            assert client == mock_client
            
            # Second call should return same client
            client2 = asyncio.run(handler.client)
            assert client2 == mock_client
            assert mock_create_client.call_count == 1
    
    @patch('app.db.base.get_settings')
    @patch('app.db.base.create_client')
    def test_health_check_success(self, mock_create_client, mock_get_settings, mock_settings):
        """Test successful health check"""
        mock_get_settings.return_value = mock_settings
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        handler = SupabaseDBHandler()
        
        with patch.object(handler, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = asyncio.run(handler.health_check())
            assert result == True
    
    @patch('app.db.base.get_settings')
    def test_health_check_failure(self, mock_get_settings, mock_settings):
        """Test health check failure"""
        mock_get_settings.return_value = mock_settings
        handler = SupabaseDBHandler()
        
        with patch.object(handler, 'get_connection') as mock_get_conn:
            mock_get_conn.side_effect = Exception("Connection failed")
            
            result = asyncio.run(handler.health_check())
            assert result == False
    
    @patch('app.db.base.get_settings')
    def test_close_connections(self, mock_get_settings, mock_settings):
        """Test closing all connections"""
        mock_get_settings.return_value = mock_settings
        handler = SupabaseDBHandler()
        
        # Add some mock connections to pool
        handler._connection_pool = {
            "conn1": {"client": MagicMock(), "created_at": time.time()},
            "conn2": {"client": MagicMock(), "created_at": time.time()}
        }
        handler._client = MagicMock()
        
        asyncio.run(handler.close())
        
        assert len(handler._connection_pool) == 0
        assert handler._client is None
    
    @patch('app.db.base.get_settings')
    def test_get_connection_stats(self, mock_get_settings, mock_settings):
        """Test connection statistics"""
        mock_get_settings.return_value = mock_settings
        handler = SupabaseDBHandler()
        
        # Add some mock connections
        handler._connection_pool = {
            "conn1": {"client": MagicMock()},
            "conn2": {"client": MagicMock()}
        }
        
        stats = asyncio.run(handler.get_connection_stats())
        
        assert stats["active_connections"] == 2
        assert stats["max_connections"] == handler._max_connections
        assert stats["connection_timeout"] == handler._connection_timeout


class TestDatabaseDependency:
    """Test database dependency injection"""
    
    @patch('app.db.base.SupabaseDBHandler')
    def test_get_db_handler_dependency(self, mock_handler_class):
        """Test database handler dependency"""
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        
        result = asyncio.run(get_db_handler())
        
        assert result == mock_handler
        mock_handler_class.assert_called_once()


class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    @patch('app.db.base.get_settings')
    @patch('app.db.base.create_client')
    def test_multiple_connections(self, mock_create_client, mock_get_settings, mock_settings):
        """Test handling multiple connections"""
        mock_get_settings.return_value = mock_settings
        mock_clients = [MagicMock() for _ in range(3)]
        mock_create_client.side_effect = mock_clients
        
        handler = SupabaseDBHandler()
        
        with patch.object(handler, '_test_connection', new_callable=AsyncMock):
            async def test_multiple():
                # Create multiple connections concurrently
                tasks = []
                for i in range(3):
                    async def use_connection():
                        async with handler.get_connection() as client:
                            return client
                    
                    tasks.append(use_connection())
                
                results = await asyncio.gather(*tasks)
                
                # All connections should be returned
                assert len(results) == 3
                assert all(result in mock_clients for result in results)
            
            asyncio.run(test_multiple())
    
    @patch('app.db.base.get_settings')
    def test_connection_cleanup_on_error(self, mock_get_settings, mock_settings):
        """Test that connections are cleaned up on errors"""
        mock_get_settings.return_value = mock_settings
        handler = SupabaseDBHandler()
        
        with patch.object(handler, '_create_client', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Database error")
            
            async def test_cleanup():
                initial_pool_size = len(handler._connection_pool)
                
                with pytest.raises(DatabaseException):
                    async with handler.get_connection():
                        pass
                
                # Pool should remain unchanged after error
                assert len(handler._connection_pool) == initial_pool_size
            
            asyncio.run(test_cleanup())
