import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr
from app.core.config import get_settings
from app.services import auth_service
from app.services.auth_service import EmailTaken, GoogleAuthError
from app.utils.limiter import limiter
from app.utils.logger import get_logger_instance
from app.utils.exceptions import (
    AuthenticationException,
    ValidationException,
    DatabaseException,
    DuplicateRecordException,
)
from app.utils.auth_middleware import jwt_handler, get_current_user
from app.utils.audit_logging import audit_logger, AuditEventType


router = APIRouter()

# Use the custom async logger instance
logger = get_logger_instance("user_auth")

# Per client address; register, login and Google sign-in share the same limit.
AUTH_RATE_LIMIT = "10/minute"
MIN_PASSWORD_CHARS = 8
MAX_PASSWORD_CHARS = 1024
MAX_ID_TOKEN_CHARS = 4096  # Google ID tokens are about 1 KB

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class GoogleLogin(BaseModel):
    id_token: str

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user_id: str

def _tokens(user: dict, issued_at: float | None = None) -> dict:
    """The app's own access and refresh tokens for an account; every sign-in route returns this body.
    `issued_at` is the time taken before the account was read (see JWTHandler._encode)."""
    token_data = {
        "sub": user["id"],
        "email": user["email"],
        "role": "user",
        "permissions": []
    }
    return {
        "access_token": jwt_handler.create_access_token(token_data, issued_at),
        "refresh_token": jwt_handler.create_refresh_token(token_data, issued_at),
        "token_type": "bearer",
        "expires_in": jwt_handler.access_token_expire_minutes * 60,
        "user_id": user["id"]
    }

async def _audit(request: Request, event_type: AuditEventType, risk_level: str, user_id: str | None = None, **details):
    # Never the email, password or token: only the outcome and the account id.
    await audit_logger.log_event(
        event_type=event_type,
        user_id=user_id,
        ip_address=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        risk_level=risk_level,
        details=details
    )

@router.post('/auth/register', response_model=LoginResponse)
@limiter.limit(AUTH_RATE_LIMIT)
async def register_user(user: UserCreate, request: Request):
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    await logger.info(f"Register attempt [ID: {correlation_id}]")
    if not MIN_PASSWORD_CHARS <= len(user.password) <= MAX_PASSWORD_CHARS:
        raise ValidationException("Password does not meet security requirements")

    try:
        account = await auth_service.register(user.email, user.password)
    except EmailTaken:
        await _audit(request, AuditEventType.AUTHENTICATION_FAILED, "medium", reason="registration_duplicate")
        raise DuplicateRecordException("User") from None
    except Exception as e:
        await logger.error(f"Registration error - {type(e).__name__} [ID: {correlation_id}]")
        raise DatabaseException("Registration failed due to database error") from None

    await _audit(request, AuditEventType.USER_CREATED, "low", user_id=account["id"])
    await logger.info(f"Registration successful [ID: {correlation_id}]")
    return _tokens(account)

@router.post('/auth/login', response_model=LoginResponse)
@limiter.limit(AUTH_RATE_LIMIT)
async def login_user(user_login: UserLogin, request: Request):
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    await logger.info(f"Login attempt [ID: {correlation_id}]")

    issued_at = time.time()  # before the hash is read: a Google link committing during argon2 revokes these tokens
    try:
        account = await auth_service.authenticate(user_login.email, user_login.password)
    except Exception as e:
        await logger.error(f"Login error - {type(e).__name__} [ID: {correlation_id}]")
        raise DatabaseException("Authentication service temporarily unavailable") from None

    if account is None:
        # Unknown email, wrong password and Google-only accounts get the same answer.
        await _audit(request, AuditEventType.AUTHENTICATION_FAILED, "medium", reason="invalid_credentials")
        raise AuthenticationException("Invalid email or password")

    await _audit(request, AuditEventType.USER_LOGIN, "low", user_id=account["id"])
    await logger.info(f"Login successful [ID: {correlation_id}]")
    return _tokens(account, issued_at)

@router.post('/auth/google', response_model=LoginResponse)
@limiter.limit(AUTH_RATE_LIMIT)
async def login_with_google(body: GoogleLogin, request: Request):
    """Sign in with a Google ID token obtained by the client; returns the same tokens as login."""
    if not get_settings().get_google_client_ids():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")  # Google sign-in is off
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    await logger.info(f"Google sign-in attempt [ID: {correlation_id}]")

    try:
        if len(body.id_token) > MAX_ID_TOKEN_CHARS:
            raise GoogleAuthError()
        account = await auth_service.login_with_google(body.id_token)
    except GoogleAuthError:
        await _audit(request, AuditEventType.AUTHENTICATION_FAILED, "medium", reason="google_token_rejected")
        raise AuthenticationException("Google sign-in failed") from None
    except Exception as e:
        await logger.error(f"Google sign-in error - {type(e).__name__} [ID: {correlation_id}]")
        raise DatabaseException("Authentication service temporarily unavailable") from None

    await _audit(request, AuditEventType.USER_LOGIN, "low", user_id=account["id"], method="google")
    await logger.info(f"Google sign-in successful [ID: {correlation_id}]")
    return _tokens(account)

@router.post('/auth/logout')
async def logout_user(current_user: dict = Depends(get_current_user)):
    # The bearer token (header only, never the URL) is validated like any protected route. App tokens are
    # stateless, so there is nothing to revoke: the client discards them.
    await logger.info("Logout successful")
    return {"message": "Logged out successfully."}
