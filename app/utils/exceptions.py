from fastapi import HTTPException, status
from typing import Any, Dict, Optional
import uuid
from datetime import datetime
from enum import Enum

class ErrorCode(str, Enum):
    """Standard error codes for the application"""
    # Authentication & Authorization
    AUTHENTICATION_FAILED = "AUTH_001"
    INVALID_TOKEN = "AUTH_002"
    TOKEN_EXPIRED = "AUTH_003"
    INSUFFICIENT_PERMISSIONS = "AUTH_004"
    
    # Validation Errors
    INVALID_INPUT = "VAL_001"
    MISSING_REQUIRED_FIELD = "VAL_002"
    INVALID_FORMAT = "VAL_003"
    SECURITY_VIOLATION = "VAL_004"
    
    # Database Errors
    DATABASE_CONNECTION_FAILED = "DB_001"
    RECORD_NOT_FOUND = "DB_002"
    DUPLICATE_RECORD = "DB_003"
    DATABASE_TIMEOUT = "DB_004"
    
    # External Service Errors
    EXTERNAL_SERVICE_UNAVAILABLE = "EXT_001"
    EXTERNAL_SERVICE_TIMEOUT = "EXT_002"
    RATE_LIMIT_EXCEEDED = "EXT_003"
    
    # Business Logic Errors
    INVALID_OPERATION = "BIZ_001"
    RESOURCE_CONFLICT = "BIZ_002"
    QUOTA_EXCEEDED = "BIZ_003"
    
    # System Errors
    INTERNAL_SERVER_ERROR = "SYS_001"
    SERVICE_UNAVAILABLE = "SYS_002"
    CONFIGURATION_ERROR = "SYS_003"

class BaseAPIException(HTTPException):
    """Base exception class for API errors with structured error handling"""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        self.error_code = error_code
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        self.details = details or {}
        
        # Safe message for client (no internal details exposed)
        safe_message = self._get_safe_message(message)
        
        detail = {
            "error_code": error_code.value,
            "message": safe_message,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            **self.details
        }
        
        super().__init__(status_code=status_code, detail=detail)
    
    def _get_safe_message(self, message: str) -> str:
        """Return a safe message that doesn't expose internal details"""
        # Override in production to return generic messages for certain error types
        return message

# Authentication Exceptions
class AuthenticationException(BaseAPIException):
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )

class InvalidTokenException(BaseAPIException):
    def __init__(self, message: str = "Invalid or malformed token", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.INVALID_TOKEN,
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )

class TokenExpiredException(BaseAPIException):
    def __init__(self, message: str = "Token has expired", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.TOKEN_EXPIRED,
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )

class InsufficientPermissionsException(BaseAPIException):
    def __init__(self, message: str = "Insufficient permissions", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.INSUFFICIENT_PERMISSIONS,
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )

# Validation Exceptions
class ValidationException(BaseAPIException):
    def __init__(self, message: str = "Invalid input provided", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.INVALID_INPUT,
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )

class SecurityViolationException(BaseAPIException):
    def __init__(self, message: str = "Security violation detected", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.SECURITY_VIOLATION,
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )

# Database Exceptions
class DatabaseException(BaseAPIException):
    def __init__(self, message: str = "Database operation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.DATABASE_CONNECTION_FAILED,
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

class RecordNotFoundException(BaseAPIException):
    def __init__(self, resource: str = "Resource", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.RECORD_NOT_FOUND,
            message=f"{resource} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )

class DuplicateRecordException(BaseAPIException):
    def __init__(self, resource: str = "Resource", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.DUPLICATE_RECORD,
            message=f"{resource} already exists",
            status_code=status.HTTP_409_CONFLICT,
            details=details
        )

# External Service Exceptions
class ExternalServiceException(BaseAPIException):
    def __init__(self, service: str, message: str = "External service error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE,
            message=f"{service}: {message}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )

class RateLimitExceededException(BaseAPIException):
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["retry_after"] = retry_after
        super().__init__(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )

# Business Logic Exceptions
class BusinessLogicException(BaseAPIException):
    def __init__(self, message: str = "Invalid operation", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.INVALID_OPERATION,
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )

class ResourceConflictException(BaseAPIException):
    def __init__(self, message: str = "Resource conflict", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.RESOURCE_CONFLICT,
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            details=details
        )

# System Exceptions
class SystemException(BaseAPIException):
    def __init__(self, message: str = "Internal system error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

class ConfigurationException(BaseAPIException):
    def __init__(self, message: str = "System configuration error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.CONFIGURATION_ERROR,
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )
