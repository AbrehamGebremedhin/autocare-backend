"""
Unit tests for custom exception classes and error handling.
"""
import pytest
from fastapi import status
import uuid
from datetime import datetime

from app.utils.exceptions import (
    ErrorCode,
    BaseAPIException,
    AuthenticationException,
    InsufficientPermissionsException,
    ValidationException,
    DatabaseException,
    RecordNotFoundException,
    DuplicateRecordException,
    RateLimitExceededException,
    BusinessLogicException,
    ResourceConflictException
)


class TestErrorCode:
    """Test error code enumeration"""
    
    def test_error_codes_exist(self):
        """Test that all expected error codes exist"""
        # Authentication codes
        assert ErrorCode.AUTHENTICATION_FAILED == "AUTH_001"
        assert ErrorCode.INVALID_TOKEN == "AUTH_002"
        assert ErrorCode.TOKEN_EXPIRED == "AUTH_003"
        assert ErrorCode.INSUFFICIENT_PERMISSIONS == "AUTH_004"
        
        # Validation codes
        assert ErrorCode.INVALID_INPUT == "VAL_001"
        assert ErrorCode.MISSING_REQUIRED_FIELD == "VAL_002"
        assert ErrorCode.INVALID_FORMAT == "VAL_003"
        assert ErrorCode.SECURITY_VIOLATION == "VAL_004"
        
        # Database codes
        assert ErrorCode.DATABASE_CONNECTION_FAILED == "DB_001"
        assert ErrorCode.RECORD_NOT_FOUND == "DB_002"
        assert ErrorCode.DUPLICATE_RECORD == "DB_003"
        assert ErrorCode.DATABASE_TIMEOUT == "DB_004"
        
        # External service codes
        assert ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE == "EXT_001"
        assert ErrorCode.EXTERNAL_SERVICE_TIMEOUT == "EXT_002"
        assert ErrorCode.RATE_LIMIT_EXCEEDED == "EXT_003"
        
        # Business logic codes
        assert ErrorCode.INVALID_OPERATION == "BIZ_001"
        assert ErrorCode.RESOURCE_CONFLICT == "BIZ_002"
        assert ErrorCode.QUOTA_EXCEEDED == "BIZ_003"
        
        # System codes
        assert ErrorCode.INTERNAL_SERVER_ERROR == "SYS_001"
        assert ErrorCode.SERVICE_UNAVAILABLE == "SYS_002"
        assert ErrorCode.CONFIGURATION_ERROR == "SYS_003"


class TestBaseAPIException:
    """Test base API exception class"""
    
    def test_basic_exception_creation(self):
        """Test basic exception creation"""
        exc = BaseAPIException(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="Test error message",
            status_code=500
        )
        
        assert exc.error_code == ErrorCode.INTERNAL_SERVER_ERROR
        assert exc.status_code == 500
        assert exc.correlation_id is not None
        assert exc.timestamp is not None
        
        # Check detail structure
        detail = exc.detail
        assert detail["error_code"] == "SYS_001"
        assert detail["message"] == "Test error message"
        assert detail["correlation_id"] == exc.correlation_id
        assert detail["timestamp"] == exc.timestamp
    
    def test_exception_with_custom_correlation_id(self):
        """Test exception with custom correlation ID"""
        custom_id = "custom-correlation-123"
        exc = BaseAPIException(
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            message="Auth failed",
            correlation_id=custom_id
        )
        
        assert exc.correlation_id == custom_id
        assert exc.detail["correlation_id"] == custom_id
    
    def test_exception_with_details(self):
        """Test exception with additional details"""
        details = {"user_id": "123", "endpoint": "/api/test"}
        exc = BaseAPIException(
            error_code=ErrorCode.INVALID_INPUT,
            message="Validation failed",
            details=details
        )
        
        # Details should be merged into the detail dict
        assert exc.detail["user_id"] == "123"
        assert exc.detail["endpoint"] == "/api/test"
    
class TestAuthenticationExceptions:
    """Test authentication-related exceptions"""
    
    def test_authentication_exception(self):
        """Test basic authentication exception"""
        exc = AuthenticationException("Invalid credentials")
        
        assert exc.error_code == ErrorCode.AUTHENTICATION_FAILED
        assert exc.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc.detail["message"] == "Invalid credentials"
        assert exc.detail["error_code"] == "AUTH_001"
    
    def test_authentication_exception_default_message(self):
        """Test authentication exception with default message"""
        exc = AuthenticationException()
        
        assert exc.detail["message"] == "Authentication failed"
    
    def test_insufficient_permissions_exception(self):
        """Test insufficient permissions exception"""
        exc = InsufficientPermissionsException("Need admin role")
        
        assert exc.error_code == ErrorCode.INSUFFICIENT_PERMISSIONS
        assert exc.status_code == status.HTTP_403_FORBIDDEN
        assert exc.detail["message"] == "Need admin role"


class TestValidationExceptions:
    """Test validation-related exceptions"""
    
    def test_validation_exception(self):
        """Test validation exception"""
        exc = ValidationException("Invalid email format")
        
        assert exc.error_code == ErrorCode.INVALID_INPUT
        assert exc.status_code == status.HTTP_400_BAD_REQUEST
        assert exc.detail["message"] == "Invalid email format"
    
class TestDatabaseExceptions:
    """Test database-related exceptions"""
    
    def test_database_exception(self):
        """Test database exception"""
        exc = DatabaseException("Connection timeout")
        
        assert exc.error_code == ErrorCode.DATABASE_CONNECTION_FAILED
        assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert exc.detail["message"] == "Connection timeout"
    
    def test_record_not_found_exception(self):
        """Test record not found exception"""
        exc = RecordNotFoundException("User")
        
        assert exc.error_code == ErrorCode.RECORD_NOT_FOUND
        assert exc.status_code == status.HTTP_404_NOT_FOUND
        assert exc.detail["message"] == "User not found"
    
    def test_record_not_found_exception_with_details(self):
        """Test record not found exception with details"""
        details = {"user_id": "123", "search_criteria": "email"}
        exc = RecordNotFoundException("User", details=details)
        
        assert exc.detail["user_id"] == "123"
        assert exc.detail["search_criteria"] == "email"
    
    def test_duplicate_record_exception(self):
        """Test duplicate record exception"""
        exc = DuplicateRecordException("Email")
        
        assert exc.error_code == ErrorCode.DUPLICATE_RECORD
        assert exc.status_code == status.HTTP_409_CONFLICT
        assert exc.detail["message"] == "Email already exists"


class TestExternalServiceExceptions:
    """Test external service-related exceptions"""
    
    def test_rate_limit_exceeded_exception(self):
        """Test rate limit exceeded exception"""
        exc = RateLimitExceededException("Too many requests", retry_after=120)
        
        assert exc.error_code == ErrorCode.RATE_LIMIT_EXCEEDED
        assert exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert exc.detail["retry_after"] == 120
    
    def test_rate_limit_exception_default_retry(self):
        """Test rate limit exception with default retry time"""
        exc = RateLimitExceededException()
        
        assert exc.detail["retry_after"] == 60  # Default value


class TestBusinessLogicExceptions:
    """Test business logic-related exceptions"""
    
    def test_business_logic_exception(self):
        """Test business logic exception"""
        exc = BusinessLogicException("Cannot delete active user")
        
        assert exc.error_code == ErrorCode.INVALID_OPERATION
        assert exc.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert exc.detail["message"] == "Cannot delete active user"
    
    def test_resource_conflict_exception(self):
        """Test resource conflict exception"""
        exc = ResourceConflictException("User is already in this group")
        
        assert exc.error_code == ErrorCode.RESOURCE_CONFLICT
        assert exc.status_code == status.HTTP_409_CONFLICT


class TestExceptionIntegration:
    """Test exception integration with FastAPI"""
    
    def test_exception_is_http_exception(self):
        """Test that custom exceptions inherit from HTTPException"""
        from fastapi import HTTPException
        
        exc = AuthenticationException("Test")
        assert isinstance(exc, HTTPException)
    
    def test_exception_preserves_fastapi_interface(self):
        """Test that exceptions work with FastAPI error handling"""
        exc = ValidationException("Invalid input")
        
        # These properties should be available for FastAPI
        assert hasattr(exc, 'status_code')
        assert hasattr(exc, 'detail')
        assert hasattr(exc, 'headers')
    
    def test_correlation_id_format(self):
        """Test correlation ID format"""
        exc = BaseAPIException(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="Test"
        )
        
        # Should be a valid UUID format
        correlation_id = exc.correlation_id
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0
        
        # Try to parse as UUID (will raise ValueError if invalid)
        try:
            uuid.UUID(correlation_id)
        except ValueError:
            pytest.fail("Correlation ID should be a valid UUID")
    
    def test_timestamp_format(self):
        """Test timestamp format"""
        exc = BaseAPIException(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="Test"
        )
        
        # Should be ISO format
        timestamp = exc.timestamp
        assert isinstance(timestamp, str)
        
        # Try to parse as datetime (will raise ValueError if invalid)
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail("Timestamp should be in ISO format")
