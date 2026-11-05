from fastapi import APIRouter, HTTPException, Depends
from app.services.user_service import user_service
from app.services.user_profile_service import UserProfileService
from app.CRUD.car_crud import CarCRUD
from app.schemas.User import UserBase
from app.schemas.Car import CarBase
from typing import List
import json
import asyncio

router = APIRouter(
    prefix="/user/cars",
    tags=["Cars"],
    responses={
        404: {"description": "User or car not found."},
        400: {"description": "Bad request or car could not be created."},
        200: {"description": "Successful Response."}
    }
)

car_crud = CarCRUD()
user_profile_service = UserProfileService()

def ensure_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        return json.loads(value)
    except Exception:
        return []

@router.get(
    "/{user_id}",
    response_model=List[CarBase],
    summary="Get user's cars",
    description="Retrieve the list of cars associated with a user with full car details.",
    responses={
        200: {"description": "List of user cars with details."},
        404: {"description": "User not found."}
    }
)
async def get_user_cars(user_id: str):
    """
    Retrieve the list of cars associated with a user with full details.
    - **user_id**: The unique identifier of the user.
    - **Returns**: List of car objects.
    """
    if not await user_service.user_exists(user_id):
        raise HTTPException(status_code=404, detail="User not found.")
    
    return await user_profile_service.get_user_cars(user_id)

@router.post(
    "/{user_id}",
    response_model=dict,
    summary="Add a car to user",
    description="Add a car to the user's list. Car will be created if it doesn't exist. Ensures car uniqueness.",
    responses={
        200: {"description": "Car added to user."},
        400: {"description": "Car could not be created."},
        404: {"description": "User not found."}
    }
)
async def add_car_to_user(user_id: str, car: dict):
    """
    Add a car to the user's list. Car will be created if it doesn't exist.
    - **user_id**: The unique identifier of the user.
    - **car**: Car data (must include make, model, year).
    - **Returns**: The car object.
    """
    if not await user_service.user_exists(user_id):
        raise HTTPException(status_code=404, detail="User not found.")
    
    # Validate required fields
    make = car.get("make")
    model = car.get("model")
    year = car.get("year")
    
    if not (make and model and year):
        raise HTTPException(status_code=400, detail="Car must include make, model, and year.")
    
    try:
        # Use the user profile service to add car (handles uniqueness)
        result = await user_profile_service.add_car_to_user(
            user_id=user_id,
            make=make,
            model=model,
            year=year
        )
        
        if result["success"]:
            return {
                **result["car_info"],
                "message": f"{make} {model} {year} successfully added",
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to add car"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add car: {str(e)}")

@router.delete(
    "/{user_id}/{car_id}",
    response_model=dict,
    summary="Remove a car from user",
    description="Remove a car from the user's list of cars.",
    responses={
        200: {"description": "Car removed from user."},
        404: {"description": "User not found."}
    }
)
async def remove_car_from_user(user_id: str, car_id: str):
    """
    Remove a car from the user's list of cars.
    - **user_id**: The unique identifier of the user.
    - **car_id**: The unique identifier of the car to remove.
    - **Returns**: The removed car ID and the updated list of car IDs.
    """
    if not await user_service.user_exists(user_id):
        raise HTTPException(status_code=404, detail="User not found.")
    
    # Remove car from user's profile
    await user_service.remove_car_from_user(user_id, car_id)
    
    # Get updated car list
    cars = await user_service.get_user_cars(user_id)
    return {"removed": car_id, "cars": cars}
