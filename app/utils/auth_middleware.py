from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timedelta
import os
from app.core.config import get_settings
from app.utils.logger import get_logger_instance
from app.utils.audit_logging import audit_logger, AuditEventType
from app.db.base import SupabaseDBHandler
import asyncio

# Security scheme
security = HTTPBearer(auto_error=False)
logger = get_logger_instance("AuthMiddleware")

class AuthenticationError(HTTPException):
    """Custom authentication error"""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

class AuthorizationError(HTTPException):
    """Custom authorization error"""
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )

class JWTHandler:
    """JWT token handling utilities"""
    
    def __init__(self):
        self.settings = get_settings()
        # In production, use a proper secret key management system
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except (jwt.PyJWTError, jwt.DecodeError, jwt.InvalidSignatureError):
            raise AuthenticationError("Invalid token")

# Global JWT handler instance
jwt_handler = JWTHandler()

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token
    """
    if not credentials:
        await audit_logger.log_event(
            event_type=AuditEventType.AUTHENTICATION_FAILED,
            ip_address=request.client.host if request.client else None,
            endpoint=request.url.path,
            method=request.method,
            risk_level="medium",
            details={"reason": "missing_token"}
        )
        raise AuthenticationError("Missing authentication token")
    
    try:
        payload = jwt_handler.verify_token(credentials.credentials)
        
        # Validate token type
        if payload.get("type") != "access":
            raise AuthenticationError("Invalid token type")
        
        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Invalid token payload")
        
        # Additional validation with Supabase
        await validate_user_with_supabase(user_id, credentials.credentials)
        
        # Store user info in request state for easy access
        request.state.current_user = {
            "id": user_id,
            "email": payload.get("email"),
            "role": payload.get("role", "user"),
            "permissions": payload.get("permissions", [])
        }
        
        return request.state.current_user
    
    except AuthenticationError:
        # Log failed authentication
        await audit_logger.log_event(
            event_type=AuditEventType.AUTHENTICATION_FAILED,
            ip_address=request.client.host if request.client else None,
            endpoint=request.url.path,
            method=request.method,
            risk_level="medium",
            details={"reason": "invalid_token"}
        )
        raise
    except Exception as e:
        await logger.error(f"Authentication error: {str(e)}")
        raise AuthenticationError("Authentication failed")

async def validate_user_with_supabase(user_id: str, token: str) -> bool:
    """
    Validate user token with Supabase
    """
    try:
        db_handler = SupabaseDBHandler()
        db = await db_handler.get_client()
        
        # Verify token with Supabase
        user_response = db.auth.get_user(token)
        if not user_response or not user_response.user:
            raise AuthenticationError("Invalid token with auth provider")
        
        return True
    except Exception as e:
        await logger.error(f"Supabase validation error: {str(e)}")
        raise AuthenticationError("Token validation failed")

async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current active user (additional validation if needed)
    """
    # Add additional checks here if needed (user status, etc.)
    return current_user

def require_permissions(required_permissions: list):
    """
    Decorator to require specific permissions
    """
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_permissions = current_user.get("permissions", [])
        user_role = current_user.get("role", "user")
        
        # Admin has all permissions
        if user_role == "admin":
            return current_user
        
        # Check specific permissions
        if not any(perm in user_permissions for perm in required_permissions):
            raise AuthorizationError(f"Required permissions: {', '.join(required_permissions)}")
        
        return current_user
    
    return permission_checker

def require_role(required_role: str):
    """
    Decorator to require specific role
    """
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role", "user")
        
        if user_role != required_role and user_role != "admin":
            raise AuthorizationError(f"Required role: {required_role}")
        
        return current_user
    
    return role_checker

# Optional authentication (for public endpoints that can benefit from user context)
async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, otherwise return None
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(request, credentials)
    except (AuthenticationError, AuthorizationError):
        return None
