from fastapi import HTTPException, status
from typing import Any, Dict, Optional
import uuid
from datetime import datetime, timezone
from enum import Enum
from http import HTTPStatus

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
    PAYLOAD_TOO_LARGE = "VAL_005"
    
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
    ROUTE_NOT_FOUND = "SYS_004"
    METHOD_NOT_ALLOWED = "SYS_005"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()  # naive UTC, as before


def error_body(error_code: str, message: str, **extra: Any) -> Dict[str, Any]:
    """The JSON body of every error response: error_code and message, a correlation id and a timestamp,
    then any extra fields. Messages are fixed texts, never request data."""
    return {
        "error_code": error_code,
        "message": message,
        "correlation_id": str(uuid.uuid4()),
        "timestamp": _utc_now_iso(),
        **extra,
    }


# Error codes for HTTPExceptions raised with a plain status (Starlette's router, FastAPI, app.services.media).
_STATUS_ERROR_CODES = {
    400: ErrorCode.INVALID_INPUT,
    401: ErrorCode.AUTHENTICATION_FAILED,
    403: ErrorCode.INSUFFICIENT_PERMISSIONS,
    404: ErrorCode.ROUTE_NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
    409: ErrorCode.RESOURCE_CONFLICT,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    415: ErrorCode.INVALID_FORMAT,
    422: ErrorCode.INVALID_INPUT,
    429: ErrorCode.RATE_LIMIT_EXCEEDED,
    500: ErrorCode.INTERNAL_SERVER_ERROR,
    503: ErrorCode.SERVICE_UNAVAILABLE,
}


def http_error_body(status_code: int, detail: Any) -> Dict[str, Any]:
    """error_body for an HTTPException: BaseAPIException bodies as they are, any other one gets the code of
    its status and its string detail (or the status phrase) as the message."""
    if isinstance(detail, dict) and "error_code" in detail and "message" in detail:
        return detail
    code = _STATUS_ERROR_CODES.get(status_code)
    message = detail if isinstance(detail, str) and detail else HTTPStatus(status_code).phrase
    return error_body(code.value if code else f"HTTP_{status_code}", message)


class BaseAPIException(HTTPException):
    """Base exception class for API errors with structured error handling"""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.error_code = error_code
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = _utc_now_iso()
        self.details = details or {}
        
        detail = {
            "error_code": error_code.value,
            "message": message,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            **self.details
        }
        
        super().__init__(status_code=status_code, detail=detail, headers=headers)
    
# Authentication Exceptions
class AuthenticationException(BaseAPIException):
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.AUTHENTICATION_FAILED,
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

class PayloadTooLargeException(BaseAPIException):
    def __init__(self, message: str = "Request body too large", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.PAYLOAD_TOO_LARGE,
            message=message,
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
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
class RateLimitExceededException(BaseAPIException):
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["retry_after"] = retry_after
        super().__init__(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
            headers={"Retry-After": str(retry_after)}
        )

# Business Logic Exceptions
class BusinessLogicException(BaseAPIException):
    def __init__(self, message: str = "Invalid operation", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.INVALID_OPERATION,
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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
class ServiceUnavailableException(BaseAPIException):
    def __init__(self, message: str = "Service temporarily unavailable", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.SERVICE_UNAVAILABLE,
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )
