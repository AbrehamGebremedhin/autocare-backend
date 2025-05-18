from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from app.db.base import SupabaseDBHandler
from app.schemas.User import UserBase
from app.core.config import get_settings
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter()

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    phone: str = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

@router.post('/auth/register', response_model=UserBase)
async def register_user(user: UserCreate):
    db = await SupabaseDBHandler().client
    try:
        response = db.auth.sign_up({
            "email": user.email,
            "password": user.password,
            "phone": user.phone
        })
        if hasattr(response, 'user') and response.user:
            user_data = response.user
        elif isinstance(response, dict) and response.get('user'):
            user_data = response['user']
        else:
            raise HTTPException(status_code=400, detail="Registration failed.")
        return UserBase(**user_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post('/auth/login')
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    db = await SupabaseDBHandler().client
    try:
        response = db.auth.sign_in_with_password({
            "email": form_data.username,
            "password": form_data.password
        })
        if hasattr(response, 'session') and response.session:
            return {"access_token": response.session.access_token, "token_type": "bearer"}
        elif isinstance(response, dict) and response.get('session'):
            return {"access_token": response['session']['access_token'], "token_type": "bearer"}
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@router.post('/auth/logout')
async def logout_user(token: str):
    db = await SupabaseDBHandler().client
    try:
        response = db.auth.sign_out(token)
        return {"message": "Logged out successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
