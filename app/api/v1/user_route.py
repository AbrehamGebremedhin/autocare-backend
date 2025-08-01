from fastapi import APIRouter, HTTPException, Depends
from app.services.user_service import user_service
from app.schemas.User import UserBase, UserProfile
from app.utils.auth_middleware import get_current_user
from typing import Optional, Dict, Any

router = APIRouter(
    prefix="/users",
    tags=["Users"],
    responses={
        404: {"description": "User not found."},
        400: {"description": "Bad request."},
        200: {"description": "Successful Response."}
    }
)

@router.get(
    "/{user_id}",
    response_model=UserBase,
    summary="Get user by ID",
    description="Retrieve user information by user ID."
)
async def get_user(user_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get user information by ID.
    - **user_id**: The unique identifier of the user.
    - **Returns**: User data including profile information.
    """
    user = await user_service.get_user_by_id(user_id, include_profile=True)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserBase(**user)

@router.get(
    "/{user_id}/profile",
    response_model=UserProfile,
    summary="Get user profile",
    description="Retrieve user profile data (cars, preferences, etc.)."
)
async def get_user_profile(user_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get user profile data.
    - **user_id**: The unique identifier of the user.
    - **Returns**: User profile data.
    """
    profile = await user_service.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found.")
    return UserProfile(**profile)

@router.put(
    "/{user_id}/profile",
    response_model=UserProfile,
    summary="Update user profile",
    description="Update user profile data."
)
async def update_user_profile(
    user_id: str, 
    profile_data: Dict[str, Any], 
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update user profile data.
    - **user_id**: The unique identifier of the user.
    - **profile_data**: Profile data to update.
    - **Returns**: Updated user profile data.
    """
    # Check if current user can update this profile (same user or admin)
    if current_user["id"] != user_id and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to update this profile.")
    
    try:
        updated_profile = await user_service.update_user_profile(user_id, profile_data)
        return UserProfile(**updated_profile)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get(
    "/me",
    response_model=UserBase,
    summary="Get current user",
    description="Get the current authenticated user's information."
)
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current user information.
    - **Returns**: Current user data including profile information.
    """
    user = await user_service.get_user_by_id(current_user["id"], include_profile=True)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserBase(**user)
