"""
Unit tests for User schema/model.
Tests validation, serialization, and business rules.
"""
import pytest
from pydantic import ValidationError
from datetime import datetime, timezone
from app.schemas.User import UserBase


class TestUserBase:
    """Test cases for UserBase model."""
    
    def test_user_creation_with_minimal_data(self):
        """Test creating a user with minimal required data."""
        user_data = {
            "email": "test@example.com"
        }
        user = UserBase(**user_data)
        
        assert user.email == "test@example.com"
        assert user.id is None
        assert user.created_at is None
        assert user.phone is None
        assert user.cars is None
        assert user.role is None
    
    def test_user_creation_with_full_data(self, sample_user_data):
        """Test creating a user with all fields populated."""
        user = UserBase(**sample_user_data)
        
        assert user.id == sample_user_data["id"]
        assert user.email == sample_user_data["email"]
        assert user.created_at == sample_user_data["created_at"]
        assert user.phone == sample_user_data["phone"]
        assert user.user_metadata == sample_user_data["user_metadata"]
        assert user.app_metadata == sample_user_data["app_metadata"]
        assert user.confirmed_at == sample_user_data["confirmed_at"]
        assert user.last_sign_in_at == sample_user_data["last_sign_in_at"]
        assert user.role == sample_user_data["role"]
        assert user.cars == sample_user_data["cars"]
    
    def test_invalid_email_raises_validation_error(self):
        """Test that invalid email raises ValidationError."""
        invalid_emails = [
            "invalid-email",
            "test@",
            "@example.com",
            "test.example.com",
            "",
            None
        ]
        
        for invalid_email in invalid_emails:
            with pytest.raises(ValidationError) as exc_info:
                UserBase(email=invalid_email)
            
            assert "email" in str(exc_info.value).lower()
    
    def test_valid_email_formats(self):
        """Test that various valid email formats are accepted."""
        valid_emails = [
            "test@example.com",
            "user.name@example.com",
            "user+tag@example.com",
            "user123@example-domain.com",
            "test.email@sub.domain.com"
        ]
        
        for valid_email in valid_emails:
            user = UserBase(email=valid_email)
            assert user.email == valid_email
    
    def test_cars_list_validation(self):
        """Test that cars field accepts list of strings."""
        user_data = {
            "email": "test@example.com",
            "cars": ["car-1", "car-2", "car-3"]
        }
        user = UserBase(**user_data)
        
        assert user.cars == ["car-1", "car-2", "car-3"]
        assert len(user.cars) == 3
    
    def test_empty_cars_list(self):
        """Test that empty cars list is handled correctly."""
        user_data = {
            "email": "test@example.com",
            "cars": []
        }
        user = UserBase(**user_data)
        
        assert user.cars == []
        assert len(user.cars) == 0
    
    def test_datetime_fields_validation(self):
        """Test that datetime fields are properly validated."""
        now = datetime.now(timezone.utc)
        user_data = {
            "email": "test@example.com",
            "created_at": now,
            "confirmed_at": now,
            "last_sign_in_at": now
        }
        user = UserBase(**user_data)
        
        assert user.created_at == now
        assert user.confirmed_at == now
        assert user.last_sign_in_at == now
    
    def test_metadata_fields_validation(self):
        """Test that metadata fields accept dictionaries."""
        user_data = {
            "email": "test@example.com",
            "user_metadata": {"name": "Test User", "preferences": {"theme": "dark"}},
            "app_metadata": {"role": "admin", "permissions": ["read", "write"]}
        }
        user = UserBase(**user_data)
        
        assert user.user_metadata["name"] == "Test User"
        assert user.user_metadata["preferences"]["theme"] == "dark"
        assert user.app_metadata["role"] == "admin"
        assert user.app_metadata["permissions"] == ["read", "write"]
    
    def test_phone_number_validation(self):
        """Test phone number field validation."""
        valid_phones = [
            "+1234567890",
            "+1-234-567-8900",
            "1234567890",
            "(123) 456-7890"
        ]
        
        for phone in valid_phones:
            user_data = {
                "email": "test@example.com",
                "phone": phone
            }
            user = UserBase(**user_data)
            assert user.phone == phone
    
    def test_role_validation(self):
        """Test role field validation."""
        valid_roles = ["user", "admin", "moderator", "premium"]
        
        for role in valid_roles:
            user_data = {
                "email": "test@example.com",
                "role": role
            }
            user = UserBase(**user_data)
            assert user.role == role
    
    def test_user_serialization(self, sample_user_data):
        """Test that user can be serialized to dict."""
        user = UserBase(**sample_user_data)
        user_dict = user.dict()
        
        assert isinstance(user_dict, dict)
        assert user_dict["email"] == sample_user_data["email"]
        assert user_dict["id"] == sample_user_data["id"]
        assert user_dict["phone"] == sample_user_data["phone"]
        assert user_dict["cars"] == sample_user_data["cars"]
    
    def test_user_json_serialization(self, sample_user_data):
        """Test that user can be serialized to JSON."""
        user = UserBase(**sample_user_data)
        user_json = user.json()
        
        assert isinstance(user_json, str)
        
        # Parse back to verify
        import json
        parsed = json.loads(user_json)
        assert parsed["email"] == sample_user_data["email"]
        assert parsed["id"] == sample_user_data["id"]
    
    def test_user_config_from_attributes(self):
        """Test that Config.from_attributes is set correctly."""
        assert UserBase.Config.from_attributes is True
    
    def test_user_update_with_new_data(self, sample_user_data):
        """Test updating user with new data."""
        user = UserBase(**sample_user_data)
        
        # Test updating individual fields
        updated_data = {
            "email": "updated@example.com",
            "phone": "+0987654321",
            "role": "premium"
        }
        
        updated_user = user.copy(update=updated_data)
        
        assert updated_user.email == "updated@example.com"
        assert updated_user.phone == "+0987654321"
        assert updated_user.role == "premium"
        # Original data should remain unchanged
        assert user.email == sample_user_data["email"]
    
    def test_user_equality_comparison(self):
        """Test user equality comparison."""
        user_data = {
            "email": "test@example.com",
            "id": "user-123"
        }
        
        user1 = UserBase(**user_data)
        user2 = UserBase(**user_data)
        
        assert user1.dict() == user2.dict()
    
    def test_user_with_none_values(self):
        """Test user creation with explicit None values."""
        user_data = {
            "email": "test@example.com",
            "id": None,
            "phone": None,
            "cars": None,
            "role": None,
            "user_metadata": None,
            "app_metadata": None
        }
        
        user = UserBase(**user_data)
        
        assert user.email == "test@example.com"
        assert user.id is None
        assert user.phone is None
        assert user.cars is None
        assert user.role is None
        assert user.user_metadata is None
        assert user.app_metadata is None
    
    def test_user_field_aliases(self):
        """Test that all expected fields are present."""
        user = UserBase(email="test@example.com")
        
        # Check that all expected fields exist
        expected_fields = [
            "id", "email", "created_at", "phone", "user_metadata",
            "app_metadata", "confirmed_at", "last_sign_in_at", "role", "cars"
        ]
        
        for field in expected_fields:
            assert hasattr(user, field), f"Field {field} is missing from UserBase"
    
    def test_user_validation_with_extra_fields(self):
        """Test behavior when extra fields are provided."""
        user_data = {
            "email": "test@example.com",
            "extra_field": "should_be_ignored"
        }
        
        # This should either ignore the extra field or raise an error
        # depending on the model configuration
        try:
            user = UserBase(**user_data)
            # If no error, check that extra field is not present
            assert not hasattr(user, "extra_field")
        except ValidationError:
            # If ValidationError is raised, that's also acceptable behavior
            pass
