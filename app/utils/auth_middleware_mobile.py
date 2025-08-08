from fastapi import Request
from typing import Dict, Any

# This is a replacement for the authentication middleware 
# that provides automatic authentication for mobile apps

async def get_current_user(request: Request, credentials=None) -> Dict[str, Any]:
    """
    Always returns a default user without requiring authentication.
    This is a replacement for the original get_current_user for mobile app compatibility.
    """
    # Create a default mock user
    default_user = {
        "id": "mobile-user",
        "email": "mobile@example.com",
        "role": "user",
        "permissions": ["read", "write"]
    }
    
    # Store user info in request state for easy access
    request.state.current_user = default_user
    
    return default_user

async def get_current_active_user(current_user: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Always returns the default user
    """
    if current_user is None:
        # Create a default mock user if none provided
        default_user = {
            "id": "mobile-user",
            "email": "mobile@example.com",
            "role": "user",
            "permissions": ["read", "write"]
        }
        return default_user
    return current_user

def require_permissions(required_permissions: list):
    """
    No-op permission checker for mobile apps
    """
    async def permission_checker(current_user: Dict[str, Any] = None):
        if current_user is None:
            # Create a default mock user if none provided
            default_user = {
                "id": "mobile-user",
                "email": "mobile@example.com",
                "role": "user",
                "permissions": ["read", "write"]
            }
            return default_user
        return current_user
    
    return permission_checker

def require_role(required_role: str):
    """
    No-op role checker for mobile apps
    """
    async def role_checker(current_user: Dict[str, Any] = None):
        if current_user is None:
            # Create a default mock user if none provided
            default_user = {
                "id": "mobile-user",
                "email": "mobile@example.com",
                "role": "user",
                "permissions": ["read", "write"]
            }
            return default_user
        return current_user
    
    return role_checker

async def get_optional_user(request: Request, credentials=None) -> Dict[str, Any]:
    """
    Always returns a default user
    """
    # Create a default mock user
    default_user = {
        "id": "mobile-user",
        "email": "mobile@example.com",
        "role": "user",
        "permissions": ["read", "write"]
    }
    
    return default_user
