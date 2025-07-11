"""
Unit tests for JSON utilities.
Tests serialization, deserialization, and datetime handling.
"""
import pytest
from datetime import datetime, date, timezone
from app.utils.json_utils import serialize_datetimes


class TestSerializeDatetimes:
    """Test cases for serialize_datetimes function."""
    
    def test_serialize_datetime_object(self):
        """Test serializing a datetime object."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = serialize_datetimes(dt)
        
        assert isinstance(result, str)
        assert result == "2025-01-01T12:00:00+00:00"
    
    def test_serialize_date_object(self):
        """Test serializing a date object."""
        d = date(2025, 1, 1)
        result = serialize_datetimes(d)
        
        assert isinstance(result, str)
        assert result == "2025-01-01"
    
    def test_serialize_naive_datetime(self):
        """Test serializing a naive datetime object."""
        dt = datetime(2025, 1, 1, 12, 0, 0)
        result = serialize_datetimes(dt)
        
        assert isinstance(result, str)
        assert result == "2025-01-01T12:00:00"
    
    def test_serialize_dict_with_datetime(self):
        """Test serializing a dictionary containing datetime objects."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {
            "id": "test-123",
            "name": "Test User",
            "created_at": dt,
            "updated_at": dt
        }
        
        result = serialize_datetimes(data)
        
        assert isinstance(result, dict)
        assert result["id"] == "test-123"
        assert result["name"] == "Test User"
        assert result["created_at"] == "2025-01-01T12:00:00+00:00"
        assert result["updated_at"] == "2025-01-01T12:00:00+00:00"
    
    def test_serialize_nested_dict_with_datetime(self):
        """Test serializing nested dictionaries with datetime objects."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {
            "user": {
                "id": "user-123",
                "profile": {
                    "created_at": dt,
                    "last_login": dt
                }
            },
            "timestamp": dt
        }
        
        result = serialize_datetimes(data)
        
        assert isinstance(result, dict)
        assert result["user"]["id"] == "user-123"
        assert result["user"]["profile"]["created_at"] == "2025-01-01T12:00:00+00:00"
        assert result["user"]["profile"]["last_login"] == "2025-01-01T12:00:00+00:00"
        assert result["timestamp"] == "2025-01-01T12:00:00+00:00"
    
    def test_serialize_list_with_datetime(self):
        """Test serializing a list containing datetime objects."""
        dt1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        data = [dt1, "string", 123, dt2]
        
        result = serialize_datetimes(data)
        
        assert isinstance(result, list)
        assert len(result) == 4
        assert result[0] == "2025-01-01T12:00:00+00:00"
        assert result[1] == "string"
        assert result[2] == 123
        assert result[3] == "2025-01-02T12:00:00+00:00"
    
    def test_serialize_list_of_dicts_with_datetime(self):
        """Test serializing a list of dictionaries with datetime objects."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = [
            {"id": "1", "created_at": dt},
            {"id": "2", "created_at": dt},
            {"id": "3", "created_at": dt}
        ]
        
        result = serialize_datetimes(data)
        
        assert isinstance(result, list)
        assert len(result) == 3
        for item in result:
            assert isinstance(item, dict)
            assert item["created_at"] == "2025-01-01T12:00:00+00:00"
    
    def test_serialize_mixed_data_types(self):
        """Test serializing mixed data types."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        d = date(2025, 1, 1)
        data = {
            "string": "test",
            "integer": 123,
            "float": 45.67,
            "boolean": True,
            "none": None,
            "datetime": dt,
            "date": d,
            "list": [1, 2, 3],
            "nested_dict": {"nested_datetime": dt}
        }
        
        result = serialize_datetimes(data)
        
        assert result["string"] == "test"
        assert result["integer"] == 123
        assert result["float"] == 45.67
        assert result["boolean"] is True
        assert result["none"] is None
        assert result["datetime"] == "2025-01-01T12:00:00+00:00"
        assert result["date"] == "2025-01-01"
        assert result["list"] == [1, 2, 3]
        assert result["nested_dict"]["nested_datetime"] == "2025-01-01T12:00:00+00:00"
    
    def test_serialize_empty_dict(self):
        """Test serializing an empty dictionary."""
        data = {}
        result = serialize_datetimes(data)
        
        assert isinstance(result, dict)
        assert len(result) == 0
        assert result == {}
    
    def test_serialize_empty_list(self):
        """Test serializing an empty list."""
        data = []
        result = serialize_datetimes(data)
        
        assert isinstance(result, list)
        assert len(result) == 0
        assert result == []
    
    def test_serialize_none_value(self):
        """Test serializing None value."""
        data = None
        result = serialize_datetimes(data)
        
        assert result is None
    
    def test_serialize_primitive_types(self):
        """Test serializing primitive types."""
        test_cases = [
            ("string", "string"),
            (123, 123),
            (45.67, 45.67),
            (True, True),
            (False, False)
        ]
        
        for input_val, expected in test_cases:
            result = serialize_datetimes(input_val)
            assert result == expected
    
    def test_serialize_complex_nested_structure(self):
        """Test serializing complex nested structure."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        d = date(2025, 1, 1)
        
        data = {
            "users": [
                {
                    "id": "user-1",
                    "profile": {
                        "created_at": dt,
                        "preferences": {
                            "last_updated": dt
                        }
                    },
                    "sessions": [
                        {"started_at": dt, "ended_at": dt},
                        {"started_at": dt, "ended_at": None}
                    ]
                },
                {
                    "id": "user-2",
                    "profile": {
                        "created_at": dt,
                        "birth_date": d
                    }
                }
            ],
            "metadata": {
                "generated_at": dt,
                "version": "1.0"
            }
        }
        
        result = serialize_datetimes(data)
        
        # Verify nested datetime serialization
        assert result["users"][0]["profile"]["created_at"] == "2025-01-01T12:00:00+00:00"
        assert result["users"][0]["profile"]["preferences"]["last_updated"] == "2025-01-01T12:00:00+00:00"
        assert result["users"][0]["sessions"][0]["started_at"] == "2025-01-01T12:00:00+00:00"
        assert result["users"][0]["sessions"][0]["ended_at"] == "2025-01-01T12:00:00+00:00"
        assert result["users"][0]["sessions"][1]["started_at"] == "2025-01-01T12:00:00+00:00"
        assert result["users"][0]["sessions"][1]["ended_at"] is None
        assert result["users"][1]["profile"]["created_at"] == "2025-01-01T12:00:00+00:00"
        assert result["users"][1]["profile"]["birth_date"] == "2025-01-01"
        assert result["metadata"]["generated_at"] == "2025-01-01T12:00:00+00:00"
        assert result["metadata"]["version"] == "1.0"
    
    def test_serialize_datetime_with_microseconds(self):
        """Test serializing datetime with microseconds."""
        dt = datetime(2025, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
        result = serialize_datetimes(dt)
        
        assert isinstance(result, str)
        assert "123456" in result
        assert result == "2025-01-01T12:00:00.123456+00:00"
    
    def test_serialize_preserves_original_data(self):
        """Test that serialization doesn't modify original data."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        original_data = {
            "id": "test-123",
            "created_at": dt,
            "nested": {"updated_at": dt}
        }
        
        # Create a copy to compare
        import copy
        original_copy = copy.deepcopy(original_data)
        
        # Serialize
        result = serialize_datetimes(original_data)
        
        # Verify original data is unchanged
        assert original_data == original_copy
        assert isinstance(original_data["created_at"], datetime)
        assert isinstance(original_data["nested"]["updated_at"], datetime)
        
        # Verify result is serialized
        assert isinstance(result["created_at"], str)
        assert isinstance(result["nested"]["updated_at"], str)
    
    def test_serialize_with_custom_datetime_format(self):
        """Test serializing with different datetime formats."""
        # Test various timezone scenarios
        dt_utc = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt_naive = datetime(2025, 1, 1, 12, 0, 0)
        
        result_utc = serialize_datetimes(dt_utc)
        result_naive = serialize_datetimes(dt_naive)
        
        assert "+00:00" in result_utc
        assert "+00:00" not in result_naive
        
        # Both should be valid ISO format strings
        assert len(result_utc) > len(result_naive)  # UTC has timezone info
