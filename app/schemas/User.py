from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    id: Optional[str] = Field(default=None, primary_key=True, unique=True)
    email: EmailStr
    created_at: Optional[datetime] = None
    phone: Optional[str] = None
    user_metadata: Optional[dict] = None
    app_metadata: Optional[dict] = None
    confirmed_at: Optional[datetime] = None
    last_sign_in_at: Optional[datetime] = None
    role: Optional[str] = None
    cars: Optional[List[str]] = None  # List of car IDs added by the user

    class Config:
        from_attributes = True
