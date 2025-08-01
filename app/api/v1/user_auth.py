from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, EmailStr
from app.db.base import SupabaseDBHandler, get_db_handler
from app.schemas.User import UserBase
from app.core.config import get_settings
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from app.services.user_service import user_service
from app.utils.logger import get_logger_instance
from app.utils.exceptions import (
    AuthenticationException, 
    ValidationException,
    DatabaseException,
    DuplicateRecordException,
    RecordNotFoundException
)
from app.utils.auth_middleware import jwt_handler
from app.utils.audit_logging import audit_logger, AuditEventType


router = APIRouter()

# Use the custom async logger instance
logger = get_logger_instance("user_auth")

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    phone: str = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

@router.post('/auth/register', response_model=UserBase)
async def register_user(user: UserCreate, request: Request, db_handler: SupabaseDBHandler = Depends(get_db_handler)):
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    # Use custom async logger method to ensure 'log_type' is present
    await logger.info(f"Register attempt for email: {user.email} [ID: {correlation_id}]")
    
    try:
        async with db_handler.get_connection() as db:
            response = db.auth.sign_up({
                "email": user.email,
                "password": user.password,
                "options": {
                    "data": {"phone": user.phone}
                }
            })
            # response may be sync, so do not await
            if hasattr(response, 'user') and response.user:
                user_data = response.user
            elif isinstance(response, dict) and response.get('user'):
                user_data = response['user']
            else:
                await audit_logger.log_event(
                    event_type=AuditEventType.AUTHENTICATION_FAILED,
                    ip_address=request.client.host if request.client else None,
                    endpoint=request.url.path,
                    method=request.method,
                    risk_level="medium",
                    details={"reason": "registration_failed", "email": user.email}
                )
                raise AuthenticationException("Registration failed - invalid response from auth provider")
            await audit_logger.log_event(
                event_type=AuditEventType.USER_CREATED,
                ip_address=request.client.host if request.client else None,
                endpoint=request.url.path,
                method=request.method,
                risk_level="low",
                details={"email": user.email, "user_id": getattr(user_data, 'id', None)}
            )
            await logger.info(f"Registration successful for email: {user.email} [ID: {correlation_id}]")
            return UserBase(**(user_data.model_dump() if hasattr(user_data, "model_dump") else dict(user_data)))
            
    except AuthenticationException:
        raise
    except Exception as e:
        await logger.error(f"Registration error for email: {user.email} - {str(e)} [ID: {correlation_id}]")
        
        # Check for specific error types
        error_str = str(e).lower()
        if "already exists" in error_str or "duplicate" in error_str:
            raise DuplicateRecordException("User", details={"email": user.email})
        elif "invalid" in error_str and "email" in error_str:
            raise ValidationException("Invalid email format")
        elif "password" in error_str and ("weak" in error_str or "short" in error_str):
            raise ValidationException("Password does not meet security requirements")
        else:
            raise DatabaseException("Registration failed due to database error")

@router.post('/auth/login')
async def login_user(form_data: OAuth2PasswordRequestForm = Depends(), request: Request = None, db_handler: SupabaseDBHandler = Depends(get_db_handler)):
    correlation_id = getattr(request.state, 'correlation_id', 'unknown') if request else 'unknown'
    await logger.info(f"Login attempt for email: {form_data.username} [ID: {correlation_id}]")
    
    try:
        async with db_handler.get_connection() as db:
            response = db.auth.sign_in_with_password({
                "email": form_data.username,
                "password": form_data.password
            })
            
            if hasattr(response, 'session') and response.session:
                session = response.session
                user = response.user
            elif isinstance(response, dict) and response.get('session'):
                session = response['session']
                user = response.get('user')
            else:
                await audit_logger.log_event(
                    event_type=AuditEventType.AUTHENTICATION_FAILED,
                    ip_address=request.client.host if request.client else None,
                    endpoint=request.url.path,
                    method=request.method,
                    risk_level="medium",
                    details={"reason": "invalid_credentials", "email": form_data.username}
                )
                raise AuthenticationException("Invalid email or password")
            
            # Create our own JWT token with additional claims
            token_data = {
                "sub": user.id if hasattr(user, 'id') else user.get('id'),
                "email": form_data.username,
                "role": getattr(user, 'role', user.get('role', 'user')),
                "permissions": getattr(user, 'permissions', user.get('permissions', []))
            }
            
            access_token = jwt_handler.create_access_token(token_data)
            refresh_token = jwt_handler.create_refresh_token(token_data)
            
            await audit_logger.log_event(
                event_type=AuditEventType.USER_LOGIN,
                user_id=token_data["sub"],
                ip_address=request.client.host if request.client else None,
                endpoint=request.url.path,
                method=request.method,
                risk_level="low",
                details={"email": form_data.username}
            )
            
            await logger.info(f"Login successful for email: {form_data.username} [ID: {correlation_id}]")
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": jwt_handler.access_token_expire_minutes * 60
            }
            
    except AuthenticationException:
        raise
    except Exception as e:
        await logger.error(f"Login error for email: {form_data.username} - {str(e)} [ID: {correlation_id}]")
        
        error_str = str(e).lower()
        if "invalid" in error_str and ("credentials" in error_str or "password" in error_str):
            raise AuthenticationException("Invalid email or password")
        elif "not confirmed" in error_str or "verification" in error_str:
            raise AuthenticationException("Email address not verified. Please check your email.")
        else:
            raise DatabaseException("Authentication service temporarily unavailable")

@router.post('/auth/logout')
async def logout_user(token: str, db_handler: SupabaseDBHandler = Depends(get_db_handler)):
    logger.info(f"Logout attempt with token: {token[:8]}... (truncated)")
    db = await db_handler.client
    try:
        response = db.auth.sign_out(token)
        logger.info("Logout successful.")
        return {"message": "Logged out successfully."}
    except Exception as e:
        logger.exception("Logout error.")
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/auth/confirm', operation_id="confirm_email_get")
async def confirm_email_get(request: Request, token: str = None, type: str = 'signup', email: str = None, db_handler: SupabaseDBHandler = Depends(get_db_handler)):
    """Confirm email via GET request (for email links)"""
    return await _confirm_email_logic(request, token, type, email, db_handler)

@router.post('/auth/confirm', operation_id="confirm_email_post")
async def confirm_email_post(request: Request, token: str = None, type: str = 'signup', email: str = None, db_handler: SupabaseDBHandler = Depends(get_db_handler)):
    """Confirm email via POST request"""
    return await _confirm_email_logic(request, token, type, email, db_handler)

async def _confirm_email_logic(request: Request, token: str = None, type: str = 'signup', email: str = None, db_handler: SupabaseDBHandler = None):
    logger.info(f"Email confirmation attempt for email: {email} (type: {type})")
    if not token or not email:
        try:
            data = await request.json()
            token = data.get('token')
            type = data.get('type', 'signup')
            email = data.get('email')
        except Exception:
            pass
    if not token or not email:
        logger.warning("Missing token or email for confirmation.")
        raise HTTPException(status_code=422, detail="Missing token or email for confirmation.")
    db = await db_handler.client
    try:
        response = db.auth.verify_otp({
            "email": email,
            "token": token,
            "type": type
        })
        if hasattr(response, 'user') and response.user:
            user_data = response.user
        elif isinstance(response, dict) and response.get('user'):
            user_data = response['user']
        else:
            logger.error(f"Email confirmation failed for email: {email}")
            raise HTTPException(status_code=400, detail="Email confirmation failed.")
        
        # Email confirmed successfully - user is now in Supabase auth.users
        # Ensure user profile exists in our user_profiles table  
        user_id = user_data.id if hasattr(user_data, 'id') else user_data.get('id')
        if user_id:
            try:
                await user_service.ensure_user_profile(user_id)
                logger.info(f"User profile ensured for user: {user_id}")
            except Exception as profile_error:
                logger.warning(f"Could not ensure user profile for {user_id}: {str(profile_error)}")
        
        logger.info(f"Email confirmed successfully for email: {email}")
        return {"message": "Email confirmed successfully."}
    except Exception as e:
        logger.exception(f"Email confirmation error for email: {email}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get('/auth/confirm-page', response_class=HTMLResponse)
async def confirm_page():
    return """
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Email Confirmation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 2em; }
            #result { margin-top: 2em; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>Confirming your email...</h2>
        <div id='result'>Please wait...</div>
        <script>
        function parseFragment() {
            const hash = window.location.hash.substring(1);
            const params = new URLSearchParams(hash);
            return {
                token: params.get('access_token'),
                type: params.get('type') || 'signup',
                email: params.get('email')
            };
        }
        async function confirm() {
            const { token, type, email } = parseFragment();
            if (!token || !email) {
                document.getElementById('result').innerText = 'Invalid confirmation link. Please check your email link.';
                return;
            }
            document.getElementById('result').innerText = 'Verifying...';
            try {
                const resp = await fetch('/api/v1/auth/confirm', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ token, type, email })
                });
                const data = await resp.json();
                if (resp.ok) {
                    document.getElementById('result').innerText = data.message || 'Email confirmed!';
                } else {
                    document.getElementById('result').innerText = data.detail || 'Confirmation failed.';
                }
            } catch (e) {
                document.getElementById('result').innerText = 'Error confirming email.';
            }
        }
        confirm();
        </script>
    </body>
    </html>
    """
