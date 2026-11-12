from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timedelta, timezone
import time
from app.core.config import get_settings
from app.utils.logger import get_logger_instance
from app.utils.audit_logging import audit_logger, AuditEventType
from app.utils.exceptions import ServiceUnavailableException
from app.services import auth_service

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

class JWTHandler:
    """JWT token handling utilities"""
    
    def __init__(self):
        self.settings = get_settings()
        self.secret_key = self.settings.JWT_SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def _encode(self, data: Dict[str, Any], lifetime: timedelta, token_type: str, issued_at: float | None) -> str:
        # iat keeps microseconds (a JWT NumericDate may be fractional): users.tokens_valid_after is compared
        # with it exactly, so a token minted just before a bump in the same second is still rejected.
        # Sign-in routes pass the time taken before they read the account, so a revocation that commits
        # while they run (a Google link during argon2) also covers the tokens they hand out.
        if issued_at is None:
            issued_at = time.time()
        expire = datetime.fromtimestamp(issued_at, timezone.utc) + lifetime
        return jwt.encode(
            {**data, "iat": issued_at, "exp": expire, "type": token_type}, self.secret_key, algorithm=self.algorithm
        )

    def create_access_token(self, data: Dict[str, Any], issued_at: float | None = None) -> str:
        """Create access token"""
        return self._encode(data, timedelta(minutes=self.access_token_expire_minutes), "access", issued_at)

    def create_refresh_token(self, data: Dict[str, Any], issued_at: float | None = None) -> str:
        """Create refresh token"""
        return self._encode(data, timedelta(days=self.refresh_token_expire_days), "refresh", issued_at)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired") from None
        except (jwt.PyJWTError, jwt.DecodeError, jwt.InvalidSignatureError):
            raise AuthenticationError("Invalid token") from None

# Global JWT handler instance
jwt_handler = JWTHandler()

def access_claims(token: str | None) -> Dict[str, Any]:
    """The verified payload of an access token (401 otherwise). Checks the token alone, never the database."""
    if not token:
        raise AuthenticationError("Missing authentication token")
    payload = jwt_handler.verify_token(token)
    if payload.get("type") != "access":
        raise AuthenticationError("Invalid token type")
    issued_at = payload.get("iat")
    if not payload.get("sub") or isinstance(issued_at, bool) or not isinstance(issued_at, (int, float)):
        raise AuthenticationError("Invalid token payload")
    return payload


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
        payload = access_claims(credentials.credentials)
        user_id, issued_at = payload["sub"], payload["iat"]

        # The user must still exist and the token must not predate a revocation (deleted account, Google link)
        await validate_user(user_id, issued_at)
        
        # Store user info in request state for easy access
        request.state.current_user = {
            "id": user_id,
            "email": payload.get("email"),
            "role": payload.get("role", "user"),
            "permissions": payload.get("permissions", []),
            "issued_at": issued_at,  # the token's iat; logout revokes from just after it
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
    except ServiceUnavailableException:
        raise
    except Exception as e:
        await logger.error(f"Authentication error: {type(e).__name__}")
        raise AuthenticationError("Authentication failed") from None

async def validate_user(user_id: str, issued_at: float) -> bool:
    """Check that the token's user still exists and accepts tokens issued at `issued_at` (else 401).
    If the lookup itself fails (database down) the answer is 503, so clients keep their tokens."""
    try:
        exists = await auth_service.user_exists(user_id, issued_at)
    except Exception as e:
        await logger.error(f"User validation error: {type(e).__name__}")
        raise ServiceUnavailableException() from None
    if not exists:
        raise AuthenticationError("Token validation failed")
    return True
