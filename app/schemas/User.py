from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    """User model based on Supabase auth.users structure"""
    id: Optional[str] = Field(default=None, description="User ID from Supabase auth.users")
    email: EmailStr
    created_at: Optional[datetime] = None
    phone: Optional[str] = None
    user_metadata: Optional[dict] = None
    app_metadata: Optional[dict] = None
    confirmed_at: Optional[datetime] = None
    last_sign_in_at: Optional[datetime] = None
    role: Optional[str] = None
    # Profile-specific fields (from user_profiles table)
    cars: Optional[List[str]] = None  # List of car IDs added by the user
    preferences: Optional[dict] = None

    class Config:
        from_attributes = True

class UserProfile(BaseModel):
    """User profile for additional data beyond Supabase auth.users"""
    user_id: str = Field(description="Reference to auth.users.id")
    cars: Optional[List[str]] = Field(default_factory=list)
    preferences: Optional[dict] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
