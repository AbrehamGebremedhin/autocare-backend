from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, EmailStr
from app.db.base import SupabaseDBHandler, get_db_handler
from app.schemas.User import UserBase
from app.core.config import get_settings
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from app.CRUD.user_crud import UserCRUD
import logging

router = APIRouter()
user_crud = UserCRUD()

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Optional: Add a console handler if not already configured globally
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    phone: str = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

@router.post('/auth/register', response_model=UserBase)
async def register_user(user: UserCreate, db_handler: SupabaseDBHandler = Depends(get_db_handler)):
    logger.info(f"Register attempt for email: {user.email}")
    db = await db_handler.client
    try:
        response = db.auth.sign_up({
            "email": user.email,
            "password": user.password,
            "options": {
                "data": {"phone": user.phone}  # Store phone in user_metadata
            }
        })
        if hasattr(response, 'user') and response.user:
            user_data = response.user
        elif isinstance(response, dict) and response.get('user'):
            user_data = response['user']
        else:
            logger.error(f"Registration failed for email: {user.email}")
            raise HTTPException(status_code=400, detail="Registration failed.")
        logger.info(f"Registration successful for email: {user.email}")
        return UserBase(**(user_data.model_dump() if hasattr(user_data, "model_dump") else dict(user_data)))
    except Exception as e:
        logger.exception(f"Registration error for email: {user.email}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/auth/login')
async def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db_handler: SupabaseDBHandler = Depends(get_db_handler)):
    logger.info(f"Login attempt for email: {form_data.username}")
    db = await db_handler.client
    try:
        response = db.auth.sign_in_with_password({
            "email": form_data.username,
            "password": form_data.password
        })
        if hasattr(response, 'session') and response.session:
            logger.info(f"Login successful for email: {form_data.username}")
            return {"access_token": response.session.access_token, "token_type": "bearer"}
        elif isinstance(response, dict) and response.get('session'):
            logger.info(f"Login successful for email: {form_data.username}")
            return {"access_token": response['session']['access_token'], "token_type": "bearer"}
        else:
            logger.warning(f"Invalid credentials for email: {form_data.username}")
            raise HTTPException(status_code=401, detail="Invalid credentials.")
    except Exception as e:
        logger.exception(f"Login error for email: {form_data.username}")
        raise HTTPException(status_code=401, detail=str(e))

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

@router.api_route('/auth/confirm', methods=["GET", "POST"])
async def confirm_email(request: Request, token: str = None, type: str = 'signup', email: str = None, db_handler: SupabaseDBHandler = Depends(get_db_handler)):
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
        existing = await user_crud.get_by_field(db, 'email', email)
        if not existing:
            user_dict = {
                'id': user_data.id if hasattr(user_data, 'id') else user_data.get('id'),
                'email': email,
                'created_at': user_data.created_at if hasattr(user_data, 'created_at') else user_data.get('created_at'),
                'phone': user_data.phone if hasattr(user_data, 'phone') else user_data.get('phone'),
                'user_metadata': user_data.user_metadata if hasattr(user_data, 'user_metadata') else user_data.get('user_metadata'),
                'app_metadata': user_data.app_metadata if hasattr(user_data, 'app_metadata') else user_data.get('app_metadata'),
                'confirmed_at': user_data.confirmed_at if hasattr(user_data, 'confirmed_at') else user_data.get('confirmed_at'),
                'last_sign_in_at': user_data.last_sign_in_at if hasattr(user_data, 'last_sign_in_at') else user_data.get('last_sign_in_at'),
                'role': user_data.role if hasattr(user_data, 'role') else user_data.get('role'),
                'cars': []
            }
            from app.CRUD.user_crud import serialize_datetimes
            user_dict = serialize_datetimes(user_dict)
            await user_crud.create(user_dict)
            logger.info(f"User created in custom table for email: {email}")
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
