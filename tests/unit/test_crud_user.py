"""
Unit tests for UserCRUD class.
Tests database operations, validation, and error handling.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from app.CRUD.user_crud import UserCRUD, serialize_datetimes
from datetime import datetime, timezone


class TestUserCRUD:
    """Test cases for UserCRUD class."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database instance."""
        db = MagicMock()
        table = MagicMock()
        db.table.return_value = table
        return db, table
    
    @pytest.fixture
    def user_crud(self):
        """Create a UserCRUD instance."""
        return UserCRUD()
    
    @pytest.fixture
    def sample_user_response(self):
        """Sample user response from database."""
        return {
            "id": "user-123",
            "email": "test@example.com",
            "created_at": "2025-01-01T12:00:00Z",
            "phone": "+1234567890",
            "user_metadata": {"name": "Test User"},
            "app_metadata": {"role": "user"},
            "confirmed_at": "2025-01-01T12:00:00Z",
            "last_sign_in_at": "2025-01-01T12:00:00Z",
            "role": "user",
            "cars": ["car-123", "car-456"]
        }
    
    def test_user_crud_initialization(self, user_crud):
        """Test UserCRUD initialization."""
        assert user_crud.table_name == 'User'
        assert hasattr(user_crud, 'get_db')
        assert hasattr(user_crud, 'unique_logic')
        assert hasattr(user_crud, 'get_by_field')
        assert hasattr(user_crud, 'user_id_exists')
    
    @pytest.mark.asyncio
    async def test_unique_logic(self, user_crud):
        """Test unique_logic method."""
        # Should not raise any exceptions
        await user_crud.unique_logic()
        
        # Should be able to pass arguments
        await user_crud.unique_logic("arg1", "arg2", key="value")
    
    @pytest.mark.asyncio
    async def test_get_by_field_success(self, user_crud, sample_user_response):
        """Test get_by_field method with successful result."""
        with patch.object(user_crud, 'get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_db.table.return_value = mock_table
            mock_get_db.return_value = mock_db
            
            # Configure the mock chain
            mock_table.select.return_value = mock_table
            mock_table.eq.return_value = mock_table
            mock_table.execute.return_value = MagicMock(data=[sample_user_response])
            
            result = await user_crud.get_by_field("email", "test@example.com")
            
            assert result == sample_user_response
            mock_db.table.assert_called_once_with('User')
            mock_table.select.assert_called_once_with('*')
            mock_table.eq.assert_called_once_with("email", "test@example.com")
            mock_table.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_field_no_result(self, user_crud):
        """Test get_by_field method with no results."""
        with patch.object(user_crud, 'get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_db.table.return_value = mock_table
            mock_get_db.return_value = mock_db
            
            # Configure the mock chain for empty result
            mock_table.select.return_value = mock_table
            mock_table.eq.return_value = mock_table
            mock_table.execute.return_value = MagicMock(data=[])
            
            result = await user_crud.get_by_field("email", "nonexistent@example.com")
            
            assert result is None
            mock_db.table.assert_called_once_with('User')
            mock_table.select.assert_called_once_with('*')
            mock_table.eq.assert_called_once_with("email", "nonexistent@example.com")
            mock_table.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_field_multiple_results(self, user_crud, sample_user_response):
        """Test get_by_field method with multiple results (should return first)."""
        with patch.object(user_crud, 'get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_db.table.return_value = mock_table
            mock_get_db.return_value = mock_db
            
            # Configure the mock chain for multiple results
            second_user = sample_user_response.copy()
            second_user["id"] = "user-456"
            mock_table.select.return_value = mock_table
            mock_table.eq.return_value = mock_table
            mock_table.execute.return_value = MagicMock(data=[sample_user_response, second_user])
            
            result = await user_crud.get_by_field("role", "user")
            
            assert result == sample_user_response
            assert result["id"] == "user-123"
    
    @pytest.mark.asyncio
    async def test_get_by_field_different_fields(self, user_crud, sample_user_response):
        """Test get_by_field method with different field names."""
        fields_to_test = ["id", "email", "phone", "role"]
        
        for field in fields_to_test:
            with patch.object(user_crud, 'get_db') as mock_get_db:
                mock_db = MagicMock()
                mock_table = MagicMock()
                mock_db.table.return_value = mock_table
                mock_get_db.return_value = mock_db
                    
                mock_table.select.return_value = mock_table
                mock_table.eq.return_value = mock_table
                mock_table.execute.return_value = MagicMock(data=[sample_user_response])
                
                result = await user_crud.get_by_field(field, "test_value")
                
                assert result == sample_user_response
                mock_table.eq.assert_called_once_with(field, "test_value")
    
    @pytest.mark.asyncio
    async def test_user_id_exists_true(self, user_crud):
        """Test user_id_exists method when user exists."""
        with patch.object(user_crud, 'get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_db.table.return_value = mock_table
            mock_get_db.return_value = mock_db
            
            # Configure the mock chain
            mock_table.select.return_value = mock_table
            mock_table.eq.return_value = mock_table
            mock_table.execute.return_value = MagicMock(data=[{"id": "user-123"}])
            
            result = await user_crud.user_id_exists("user-123")
            
            assert result is True
            mock_db.table.assert_called_once_with('User')
            mock_table.select.assert_called_once_with('id')
            mock_table.eq.assert_called_once_with('id', 'user-123')
            mock_table.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_user_id_exists_false(self, user_crud):
        """Test user_id_exists method when user doesn't exist."""
        with patch.object(user_crud, 'get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_db.table.return_value = mock_table
            mock_get_db.return_value = mock_db
            
            # Configure the mock chain for empty result
            mock_table.select.return_value = mock_table
            mock_table.eq.return_value = mock_table
            mock_table.execute.return_value = MagicMock(data=[])
            
            result = await user_crud.user_id_exists("nonexistent-user")
            
            assert result is False
            mock_db.table.assert_called_once_with('User')
            mock_table.select.assert_called_once_with('id')
            mock_table.eq.assert_called_once_with('id', 'nonexistent-user')
            mock_table.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_user_id_exists_none_response(self, user_crud):
        """Test user_id_exists method with None response."""
        with patch.object(user_crud, 'get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_db.table.return_value = mock_table
            mock_get_db.return_value = mock_db
            
            # Configure the mock chain for None response
            mock_table.select.return_value = mock_table
            mock_table.eq.return_value = mock_table
            mock_table.execute.return_value = MagicMock(data=None)
            
            result = await user_crud.user_id_exists("user-123")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, user_crud):
        """Test handling of database errors."""
        with patch.object(user_crud, 'get_db') as mock_get_db:
            mock_get_db.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception) as exc_info:
                await user_crud.get_by_field("email", "test@example.com")
            
            assert "Database connection failed" in str(exc_info.value)
    
    def test_serialize_datetimes_function(self):
        """Test serialize_datetimes function."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {
            "id": "user-123",
            "email": "test@example.com",
            "created_at": dt,
            "metadata": {
                "updated_at": dt,
                "nested": {
                    "timestamp": dt
                }
            },
            "sessions": [
                {"started_at": dt, "ended_at": dt}
            ]
        }
        
        result = serialize_datetimes(data)
        
        assert result["id"] == "user-123"
        assert result["email"] == "test@example.com"
        assert result["created_at"] == dt.isoformat()
        assert result["metadata"]["updated_at"] == dt.isoformat()
        assert result["metadata"]["nested"]["timestamp"] == dt.isoformat()
        assert result["sessions"][0]["started_at"] == dt.isoformat()
        assert result["sessions"][0]["ended_at"] == dt.isoformat()
    
    def test_serialize_datetimes_with_non_dict_list_items(self):
        """Test serialize_datetimes function with non-dict items in lists."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {
            "mixed_list": [
                "string_item",
                123,
                {"timestamp": dt},
                dt  # This datetime won't be serialized as it's not in a dict
            ]
        }
        
        result = serialize_datetimes(data)
        
        assert result["mixed_list"][0] == "string_item"
        assert result["mixed_list"][1] == 123
        assert result["mixed_list"][2]["timestamp"] == dt.isoformat()
        # The datetime object directly in the list is not serialized by the current implementation
        assert result["mixed_list"][3] == dt
    
    def test_serialize_datetimes_with_empty_data(self):
        """Test serialize_datetimes function with empty data."""
        empty_data = {}
        result = serialize_datetimes(empty_data)
        
        assert result == {}
        assert isinstance(result, dict)
    
    def test_serialize_datetimes_with_none_values(self):
        """Test serialize_datetimes function with None values."""
        data = {
            "id": "user-123",
            "created_at": None,
            "updated_at": None,
            "metadata": {
                "value": None
            }
        }
        
        result = serialize_datetimes(data)
        
        assert result["id"] == "user-123"
        assert result["created_at"] is None
        assert result["updated_at"] is None
        assert result["metadata"]["value"] is None
    
    def test_serialize_datetimes_immutability(self):
        """Test that serialize_datetimes modifies data in place."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        original_data = {
            "id": "user-123",
            "created_at": dt,
            "metadata": {"timestamp": dt}
        }
        
        # Serialize (modifies in place)
        result = serialize_datetimes(original_data)
        
        # Result should be the same object
        assert result is original_data
        
        # Data should be serialized
        assert isinstance(result["created_at"], str)
        assert isinstance(result["metadata"]["timestamp"], str)
        assert result["created_at"] == dt.isoformat()
        assert result["metadata"]["timestamp"] == dt.isoformat()
    
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, user_crud):
        """Test concurrent database operations."""
        with patch.object(user_crud, 'get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_db.table.return_value = mock_table
            mock_get_db.return_value = mock_db
            
            # Configure the mock chain
            mock_table.select.return_value = mock_table
            mock_table.eq.return_value = mock_table
            mock_table.execute.return_value = MagicMock(data=[{"id": "user-123"}])
            
            # Run multiple concurrent operations
            tasks = [
                user_crud.user_id_exists(f"user-{i}") 
                for i in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All operations should succeed
            assert all(result is True for result in results)
            assert len(results) == 10
    
    @pytest.mark.asyncio
    async def test_get_by_field_with_special_characters(self, user_crud, sample_user_response):
        """Test get_by_field with special characters in field values."""
        special_values = [
            "test@example.com",
            "user with spaces",
            "user-with-dashes",
            "user_with_underscores",
            "user.with.dots",
            "user+with+plus",
            "user#with#hash"
        ]
        
        for value in special_values:
            with patch.object(user_crud, 'get_db') as mock_get_db:
                mock_db = MagicMock()
                mock_table = MagicMock()
                mock_db.table.return_value = mock_table
                mock_get_db.return_value = mock_db
                    
                mock_table.select.return_value = mock_table
                mock_table.eq.return_value = mock_table
                mock_table.execute.return_value = MagicMock(data=[sample_user_response])
                
                result = await user_crud.get_by_field("email", value)
                
                assert result == sample_user_response
                mock_table.eq.assert_called_with("email", value)
    
    def test_inheritance_from_base_crud(self, user_crud):
        """Test that UserCRUD inherits from BaseCRUD."""
        from app.db.crud import BaseCRUD
        
        assert isinstance(user_crud, BaseCRUD)
        assert hasattr(user_crud, 'table_name')
        assert user_crud.table_name == 'User'
    
    @pytest.mark.asyncio
    async def test_error_handling_in_user_id_exists(self, user_crud):
        """Test error handling in user_id_exists method."""
        with patch.object(user_crud, 'get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_db.table.return_value = mock_table
            mock_get_db.return_value = mock_db
            
            # Configure the mock to raise an exception
            mock_table.select.return_value = mock_table
            mock_table.eq.return_value = mock_table
            mock_table.execute.side_effect = Exception("Database error")
            
            with pytest.raises(Exception) as exc_info:
                await user_crud.user_id_exists("user-123")
            
            assert "Database error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_by_field_with_none_value(self, user_crud):
        """Test get_by_field with None as field value."""
        with patch.object(user_crud, 'get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_db.table.return_value = mock_table
            mock_get_db.return_value = mock_db
            
            mock_table.select.return_value = mock_table
            mock_table.eq.return_value = mock_table
            mock_table.execute.return_value = MagicMock(data=[])
            
            result = await user_crud.get_by_field("phone", None)
            
            assert result is None
            mock_table.eq.assert_called_once_with("phone", None)
    
    @pytest.mark.asyncio
    async def test_user_id_exists_with_empty_string(self, user_crud):
        """Test user_id_exists with empty string."""
        with patch.object(user_crud, 'get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_db.table.return_value = mock_table
            mock_get_db.return_value = mock_db
            
            mock_table.select.return_value = mock_table
            mock_table.eq.return_value = mock_table
            mock_table.execute.return_value = MagicMock(data=[])
            
            result = await user_crud.user_id_exists("")
            
            assert result is False
            mock_table.eq.assert_called_once_with('id', '')
    
    def test_serialize_datetimes_with_custom_object(self):
        """Test serialize_datetimes with objects that have isoformat method."""
        class CustomDatetime:
            def isoformat(self):
                return "2025-01-01T12:00:00Z"
        
        custom_obj = CustomDatetime()
        data = {
            "custom_date": custom_obj,
            "regular_field": "value"
        }
        
        result = serialize_datetimes(data)
        
        assert result["custom_date"] == "2025-01-01T12:00:00Z"
        assert result["regular_field"] == "value"
    
    def test_serialize_datetimes_recursive_depth(self):
        """Test serialize_datetimes with deeply nested structures."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Create deeply nested structure
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "timestamp": dt
                            }
                        }
                    }
                }
            }
        }
        
        result = serialize_datetimes(data)
        
        assert result["level1"]["level2"]["level3"]["level4"]["level5"]["timestamp"] == dt.isoformat()
    
    import asyncio
