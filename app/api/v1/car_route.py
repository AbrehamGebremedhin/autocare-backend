from fastapi import APIRouter, HTTPException, Depends
from app.CRUD.user_crud import UserCRUD
from app.CRUD.car_crud import CarCRUD
from app.schemas.User import UserBase
from typing import List
import json

router = APIRouter(
    prefix="/user/cars",
    tags=["Cars"],
    responses={
        404: {"description": "User or car not found."},
        400: {"description": "Bad request or car could not be created."},
        200: {"description": "Successful Response."}
    }
)

user_crud = UserCRUD()
car_crud = CarCRUD()

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
    response_model=List[str],
    summary="Get user's cars",
    description="Retrieve the list of car IDs associated with a user.",
    responses={
        200: {"description": "List of car IDs."},
        404: {"description": "User not found."}
    }
)
async def get_user_cars(user_id: str):
    """
    Retrieve the list of car IDs associated with a user.
    - **user_id**: The unique identifier of the user.
    - **Returns**: List of car IDs.
    """
    user = await user_crud.read({"id": user_id})
    if not user or not user[0]:
        raise HTTPException(status_code=404, detail="User not found.")
    return ensure_list(user[0].get("cars", []))

@router.post(
    "/{user_id}",
    response_model=dict,
    summary="Add a car to user",
    description="Add a car to the user's list. If the car does not exist, it will be created. If it exists, it will only be added to the user's cars field.",
    responses={
        200: {"description": "Car added to user."},
        400: {"description": "Car could not be created."},
        404: {"description": "User not found."}
    }
)
async def add_car_to_user(user_id: str, car: dict):
    """
    Add a car to the user's list. If the car does not exist, it will be created. If it exists, it will only be added to the user's cars field.
    - **user_id**: The unique identifier of the user.
    - **car**: Car data (must include make, model, year).
    - **Returns**: The car object.
    """
    user = await user_crud.read({"id": user_id})
    if not user or not user[0]:
        raise HTTPException(status_code=404, detail="User not found.")
    # Try to get car by make, model, year (unique)
    make = car.get("make")
    model = car.get("model")
    year = car.get("year")
    car_obj = None
    if make and model and year:
        car_obj = await car_crud.get_car_by_make_model_year(make, model, year)
    if car_obj:
        car_id = car_obj.get("id")
    else:
        car_result = await car_crud.create(car)
        if not car_result or not car_result[0]:
            raise HTTPException(status_code=400, detail="Car could not be created.")
        car_obj = car_result[0]
        car_id = car_obj.get("id")
    cars = ensure_list(user[0].get("cars", []))
    if car_id not in cars:
        cars.append(car_id)
        await user_crud.update({"id": user_id}, {"cars": cars})
    return car_obj

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
    user = await user_crud.read({"id": user_id})
    if not user or not user[0]:
        raise HTTPException(status_code=404, detail="User not found.")
    cars = ensure_list(user[0].get("cars", []))
    if car_id in cars:
        cars.remove(car_id)
        await user_crud.update({"id": user_id}, {"cars": cars})
    return {"removed": car_id, "cars": cars}
