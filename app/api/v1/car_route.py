from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.schemas.Car import CarBase
from app.services.user_profile_service import UserProfileService
from app.utils.auth_middleware import get_current_user

router = APIRouter(prefix="/cars", tags=["Cars"])

user_profile_service = UserProfileService()


class CarCreate(BaseModel):
    make: str
    model: str
    year: int


@router.get("", response_model=List[CarBase], summary="List the current user's cars")
async def list_cars(current_user: dict = Depends(get_current_user)):
    return await user_profile_service.get_user_cars(current_user["id"])


@router.post("", response_model=CarBase, summary="Add a car to the current user")
async def add_car(car: CarCreate, current_user: dict = Depends(get_current_user)):
    result = await user_profile_service.add_car_to_user(
        user_id=current_user["id"], make=car.make, model=car.model, year=car.year
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail="Could not add car")
    return result["car_info"]


@router.delete("/{car_id}", summary="Remove a car from the current user")
async def delete_car(car_id: str, current_user: dict = Depends(get_current_user)):
    result = await user_profile_service.remove_car_from_user(current_user["id"], car_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail="Car not found")
    return {"deleted": car_id}
