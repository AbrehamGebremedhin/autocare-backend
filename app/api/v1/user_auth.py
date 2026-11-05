from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, EmailStr
from app.db.base import SupabaseDBHandler, get_db_handler
from app.schemas.User import UserBase
from app.core.config import get_settings
from fastapi.responses import HTMLResponse
from supabase import AuthError
from app.services.user_service import user_service
from app.utils.logger import get_logger_instance
from app.utils.exceptions import (
    AuthenticationException,
    BaseAPIException,
    ValidationException,
    DatabaseException,
    DuplicateRecordException,
    RateLimitExceededException,
)
from fastapi.security import HTTPAuthorizationCredentials
from app.utils.auth_middleware import jwt_handler, get_current_user, security
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

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user_id: str

def _auth_provider_error(e: AuthError, registering: bool) -> BaseAPIException:
    """Map a Supabase auth error to a client error by its code and status; its message is never echoed."""
    code = getattr(e, "code", None)
    status_code = getattr(e, "status", None) or 0
    if code in ("user_already_exists", "email_exists"):
        return DuplicateRecordException("User")
    if code == "weak_password":
        return ValidationException("Password does not meet security requirements")
    if code in ("email_address_invalid", "validation_failed"):
        return ValidationException("Invalid email or password format")
    if code == "email_not_confirmed":
        return AuthenticationException("Email address not verified. Please check your email.")
    if code in ("over_request_rate_limit", "over_email_send_rate_limit") or status_code == 429:
        return RateLimitExceededException("Too many attempts. Please try again later.")
    if code == "invalid_credentials" or (not registering and status_code in (400, 401)):
        return AuthenticationException("Invalid email or password")
    if 400 <= status_code < 500:
        return ValidationException("Registration was rejected") if registering else AuthenticationException()
    return DatabaseException("Authentication service temporarily unavailable")

@router.post('/auth/register', response_model=UserBase)
async def register_user(user: UserCreate, request: Request, db_handler: SupabaseDBHandler = Depends(get_db_handler)):
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    # Use custom async logger method to ensure 'log_type' is present
    await logger.info(f"Register attempt [ID: {correlation_id}]")
    
    try:
        async with db_handler.get_connection() as db:
            try:
                response = db.auth.sign_up({
                    "email": user.email,
                    "password": user.password,
                    "options": {
                        "data": {"phone": user.phone}
                    }
                })
            except AuthError as e:
                # Caught here: get_connection turns anything else raised in the block into a generic 500.
                await logger.warning(f"Registration rejected - {type(e).__name__} {getattr(e, 'code', None)} [ID: {correlation_id}]")
                raise _auth_provider_error(e, registering=True) from None
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
            await logger.info(f"Registration successful [ID: {correlation_id}]")
            return UserBase(**(user_data.model_dump() if hasattr(user_data, "model_dump") else dict(user_data)))
            
    except BaseAPIException:
        raise
    except Exception as e:
        await logger.error(f"Registration error - {type(e).__name__} [ID: {correlation_id}]")
        raise DatabaseException("Registration failed due to database error")

@router.post('/auth/login', response_model=LoginResponse)
async def login_user(user_login: UserLogin, request: Request = None, db_handler: SupabaseDBHandler = Depends(get_db_handler)):
    correlation_id = getattr(request.state, 'correlation_id', 'unknown') if request else 'unknown'
    await logger.info(f"Login attempt [ID: {correlation_id}]")
    
    try:
        async with db_handler.get_connection() as db:
            try:
                response = db.auth.sign_in_with_password({
                    "email": user_login.email,
                    "password": user_login.password
                })
            except AuthError as e:
                # Caught here: get_connection turns anything else raised in the block into a generic 500.
                await logger.warning(f"Login rejected - {type(e).__name__} {getattr(e, 'code', None)} [ID: {correlation_id}]")
                raise _auth_provider_error(e, registering=False) from None

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
                    details={"reason": "invalid_credentials", "email": user_login.email}
                )
                raise AuthenticationException("Invalid email or password")
            
            # Create our own JWT token with additional claims
            # Safely extract user ID - handle both object and dict formats
            if hasattr(user, 'id'):
                user_id = user.id
            elif isinstance(user, dict) and 'id' in user:
                user_id = user['id']
            else:
                user_id = str(user) if user else None
            
            # Ensure we have a valid user_id
            if not user_id:
                raise AuthenticationException("Failed to extract user ID from authentication response")
            
            # Safely extract role - handle both object and dict formats
            role = 'user'  # default value
            if hasattr(user, 'role'):
                role = user.role
            elif isinstance(user, dict) and 'role' in user:
                role = user['role']
            
            # Safely extract permissions - handle both object and dict formats
            permissions = []  # default value
            if hasattr(user, 'permissions'):
                permissions = user.permissions or []
            elif isinstance(user, dict) and 'permissions' in user:
                permissions = user['permissions'] or []
            
            token_data = {
                "sub": user_id,
                "email": user_login.email,
                "role": role,
                "permissions": permissions
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
                details={"email": user_login.email}
            )
            
            await logger.info(f"Login successful [ID: {correlation_id}]")
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": jwt_handler.access_token_expire_minutes * 60,
                "user_id": user_id
            }
            
    except BaseAPIException:
        raise
    except Exception as e:
        await logger.error(f"Login error - {type(e).__name__} [ID: {correlation_id}]")
        raise DatabaseException("Authentication service temporarily unavailable")

@router.post('/auth/logout')
async def logout_user(
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db_handler: SupabaseDBHandler = Depends(get_db_handler),
):
    # The token comes from the Authorization header only, never from the URL.
    await logger.info("Logout attempt")
    async with db_handler.get_connection() as db:
        try:
            db.auth.admin.sign_out(credentials.credentials)
        except Exception as e:
            await logger.error(f"Logout error: {type(e).__name__}")
            raise HTTPException(status_code=400, detail="Logout failed.")
    await logger.info("Logout successful")
    return {"message": "Logged out successfully."}

@router.get('/auth/confirm', operation_id="confirm_email_get")
async def confirm_email_get(request: Request, token: str = None, type: str = 'signup', email: str = None, db_handler: SupabaseDBHandler = Depends(get_db_handler)):
    """Confirm email via GET request (for email links) - handles URL params and redirects to page"""
    # If this is a direct GET request with parameters, redirect to the confirmation page with the params
    if token:
        # Construct the confirmation page URL with the parameters
        base_url = "http://localhost:8000/api/v1/auth/confirm-page"
        params = []
        if token:
            params.append(f"access_token={token}")
        if type and type != 'signup':
            params.append(f"type={type}")
        if email:
            params.append(f"email={email}")
        
        if params:
            redirect_url = f"{base_url}#{'&'.join(params)}"
        else:
            redirect_url = base_url
            
        # Return a redirect response
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=redirect_url, status_code=302)
    else:
        # Try to process confirmation directly
        return await _confirm_email_logic(request, token, type, email, db_handler)

@router.post('/auth/confirm', operation_id="confirm_email_post")
async def confirm_email_post(request: Request, token: str = None, type: str = 'signup', email: str = None, db_handler: SupabaseDBHandler = Depends(get_db_handler)):
    """Confirm email via POST request"""
    return await _confirm_email_logic(request, token, type, email, db_handler)

async def _confirm_email_logic(request: Request, token: str = None, token_type: str = 'signup', email: str = None, db_handler: SupabaseDBHandler = None):
    await logger.info(f"Email confirmation attempt - Initial params: token_present={bool(token)}, type={token_type}")
    
    if not token:
        try:
            data = await request.json()
            token = data.get('token')
            token_type = data.get('type', 'signup')
            email = data.get('email') or email
            await logger.info(f"Got params from request body: token_present={bool(token)}, type={token_type}")
        except Exception as e:
            await logger.warning(f"Could not parse JSON from request: {type(e).__name__}")
            pass
    
    if not token:
        await logger.warning("Missing token for confirmation.")
        raise HTTPException(status_code=422, detail="Missing token for confirmation.")
    
    async with db_handler.get_connection() as db:
        try:
            # Prepare verification data - email is optional for some confirmation types
            verification_data = {
                "token": token,
                "type": token_type
            }
            
            # Only add email if provided (some Supabase confirmation types work without it)
            if email:
                verification_data["email"] = email
                
            await logger.info(f"Attempting verification, type={token_type}, token_present=True")
            
            try:
                # For email confirmation, we might need to use exchange_code_for_session instead of verify_otp
                # Let's try verify_otp first, then fall back to other methods if needed
                response = db.auth.verify_otp(verification_data)
                await logger.info("Supabase verify_otp successful")
            except Exception as verify_error:
                await logger.error(f"verify_otp failed: {type(verify_error).__name__}")
                
                # Try alternative approach for email confirmation
                try:
                    await logger.info("Trying exchange_code_for_session as alternative...")
                    response = db.auth.exchange_code_for_session({"auth_code": token})
                    await logger.info("exchange_code_for_session successful")
                except Exception as exchange_error:
                    await logger.error(f"exchange_code_for_session also failed: {type(exchange_error).__name__}")
                    # Re-raise the original error
                    raise verify_error
            
            if hasattr(response, 'user') and response.user:
                user_data = response.user
                await logger.info("Got user from response.user")
            elif isinstance(response, dict) and response.get('user'):
                user_data = response['user']
                await logger.info("Got user from response dict")
            else:
                await logger.error("Email confirmation failed - no user returned")
                raise HTTPException(status_code=400, detail="Email confirmation failed - invalid token.")
            
            # Email confirmed successfully - user is now in Supabase auth.users
            # Ensure user profile exists in our user_profiles table  
            user_id = user_data.id if hasattr(user_data, 'id') else user_data.get('id')
            if user_id:
                try:
                    await user_service.ensure_user_profile(user_id)
                    await logger.info("User profile ensured")
                except Exception as profile_error:
                    await logger.warning(f"Could not ensure user profile: {type(profile_error).__name__}")
            
            await logger.info("Email confirmed successfully")
            return {"message": "Email confirmed successfully."}
            
        except Exception as e:
            await logger.error(f"Email confirmation error: {type(e).__name__}")
            
            # Provide more helpful error messages
            error_msg = str(e).lower()
            if "invalid" in error_msg or "expired" in error_msg:
                raise HTTPException(status_code=400, detail="The confirmation link is invalid or has expired. Please request a new confirmation email.")
            elif "already" in error_msg:
                raise HTTPException(status_code=400, detail="This email has already been confirmed.")
            elif "'str' object is not callable" in error_msg:
                raise HTTPException(status_code=400, detail="Internal error in email confirmation. Please try again or contact support.")
            else:
                raise HTTPException(status_code=400, detail="Email confirmation failed. Please request a new confirmation email.")

@router.get('/auth/confirm-page', response_class=HTMLResponse)
async def confirm_page(request: Request):
    # Get the CSP nonce from the request state (set by security middleware)
    nonce = getattr(request.state, 'csp_nonce', '')
    
    return f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email Confirmation</title>
        <style nonce="{nonce}">
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%);
                margin: 0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .container {{
                background: #fff;
                border-radius: 16px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.08);
                padding: 2.5em 2em 2em 2em;
                max-width: 400px;
                width: 100%;
                text-align: center;
                animation: fadeIn 0.8s;
            }}
            h2 {{
                margin-bottom: 1.2em;
                color: #1a365d;
                font-weight: 600;
                font-size: 1.5em;
            }}
            #result {{
                margin-top: 1.5em;
                font-weight: 500;
                font-size: 1.1em;
                min-height: 2.5em;
                transition: color 0.2s;
            }}
            .error {{
                color: #e53e3e;
                background: #fff5f5;
                border-radius: 8px;
                padding: 0.7em 1em;
                margin-top: 1em;
                border: 1px solid #fed7d7;
            }}
            .success {{
                color: #38a169;
                background: #f0fff4;
                border-radius: 8px;
                padding: 0.7em 1em;
                margin-top: 1em;
                border: 1px solid #c6f6d5;
            }}
            .loader {{
                border: 4px solid #e2e8f0;
                border-top: 4px solid #3182ce;
                border-radius: 50%;
                width: 32px;
                height: 32px;
                animation: spin 1s linear infinite;
                display: inline-block;
                vertical-align: middle;
                margin-right: 0.5em;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(30px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .redirect-msg {{
                margin-top: 1.5em;
                color: #4a5568;
                font-size: 0.98em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Email Confirmation</h2>
            <div id='result'>
                <span class="loader"></span>
                Please wait...
            </div>
            <div id="redirect" class="redirect-msg" hidden></div>
        </div>
        <script nonce="{nonce}">
        try {{
            function parseFragment() {{
                const hash = window.location.hash.substring(1);
                const search = window.location.search.substring(1);
                const hashParams = new URLSearchParams(hash);
                const queryParams = new URLSearchParams(search);
                const token = hashParams.get('access_token') ||
                              hashParams.get('token') ||
                              queryParams.get('token') ||
                              queryParams.get('access_token');
                const type = hashParams.get('type') ||
                             queryParams.get('type') ||
                             'signup';
                const email = hashParams.get('email') ||
                              queryParams.get('email');
                return {{ token, type, email }};
            }}

            function showResult(message, type) {{
                const result = document.getElementById('result');
                result.className = type;
                result.innerHTML = message;
            }}

            function showLoader(message) {{
                const result = document.getElementById('result');
                result.className = '';
                result.innerHTML = '<span class="loader"></span>' + (message || 'Please wait...');
            }}

            async function confirm() {{
                try {{
                    const {{ token, type, email }} = parseFragment();
                    if (!token) {{
                        showResult('No confirmation token found in URL. Please check your email link.', 'error');
                        return;
                    }}
                    showLoader('Verifying token...');
                    const response = await fetch('/api/v1/auth/confirm', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ token, type, email }})
                    }});
                    const data = await response.json();
                    if (response.ok) {{
                        showResult(data.message || 'Email confirmed successfully!', 'success');
                        document.getElementById('redirect').hidden = false;
                        document.getElementById('redirect').innerText = 'Redirecting to the app...';
                        setTimeout(() => {{
                            window.location.href = '/';
                        }}, 3000);
                    }} else {{
                        showResult(data.detail || 'Email confirmation successful.', 'success');
                    }}
                }} catch (e) {{
                    showResult('Error confirming email: ' + e.message, 'error');
                    console.error('Confirmation error:', e);
                }}
            }}

            confirm();
        }} catch (e) {{
            showResult('JavaScript error: ' + e.message, 'error');
            console.error('Page error:', e);
        }}
        </script>
    </body>
    </html>
    """
