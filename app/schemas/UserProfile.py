"""
User Profile schema for storing additional user data beyond Supabase auth.users
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class UserProfileBase(BaseModel):
    """Base schema for user profiles"""
    user_id: str = Field(..., description="Supabase user ID from auth.users")
    cars: Optional[List[str]] = Field(default=[], description="List of car IDs owned by user")
    preferences: Optional[Dict[str, Any]] = Field(default={}, description="User preferences and settings")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True

class UserProfileCreate(BaseModel):
    """Schema for creating user profiles"""
    cars: Optional[List[str]] = Field(default=[])
    preferences: Optional[Dict[str, Any]] = Field(default={})

class UserProfileUpdate(BaseModel):
    """Schema for updating user profiles"""
    cars: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None

class SupabaseUser(BaseModel):
    """Schema representing a Supabase auth user"""
    id: str
    email: str
    created_at: Optional[datetime] = None
    phone: Optional[str] = None
    user_metadata: Optional[Dict[str, Any]] = Field(default={})
    app_metadata: Optional[Dict[str, Any]] = Field(default={})
    confirmed_at: Optional[datetime] = None
    last_sign_in_at: Optional[datetime] = None
    role: Optional[str] = Field(default="authenticated")

    class Config:
        from_attributes = True

class UserWithProfile(SupabaseUser):
    """Combined user and profile data"""
    profile: Optional[UserProfileBase] = None

    class Config:
        from_attributes = True
