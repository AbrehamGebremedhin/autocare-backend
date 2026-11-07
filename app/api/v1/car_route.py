from datetime import date
from typing import Annotated, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path
from pydantic import BaseModel, Field, field_validator

from app.CRUD.car_crud import ingest_car_manual
from app.schemas.Car import MAX_CAR_ID_CHARS, MAX_CAR_NAME_CHARS, MIN_CAR_YEAR, NO_NUL, CarBase
from app.services.user_profile_service import UserProfileService
from app.utils.auth_middleware import get_current_user

router = APIRouter(prefix="/cars", tags=["Cars"])

user_profile_service = UserProfileService()


class CarCreate(BaseModel):
    make: str = Field(min_length=1, max_length=MAX_CAR_NAME_CHARS, pattern=NO_NUL)
    model: str = Field(min_length=1, max_length=MAX_CAR_NAME_CHARS, pattern=NO_NUL)
    year: int = Field(ge=MIN_CAR_YEAR)

    @field_validator("year")
    @classmethod
    def _not_after_next_year(cls, year: int) -> int:
        latest = date.today().year + 1  # next year's models are on sale
        if year > latest:
            raise ValueError(f"year must be at most {latest}")
        return year


@router.get("", response_model=List[CarBase], summary="List the current user's cars")
async def list_cars(current_user: dict = Depends(get_current_user)):
    return await user_profile_service.get_user_cars(current_user["id"])


@router.post("", response_model=CarBase, summary="Add a car to the current user")
async def add_car(
    car: CarCreate, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)
):
    result = await user_profile_service.add_car_to_user(
        user_id=current_user["id"], make=car.make, model=car.model, year=car.year
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail="Could not add car")
    car_info = result["car_info"]
    if car_info.get("kb_status", "pending") == "pending":
        background_tasks.add_task(ingest_car_manual, car_info)
    return car_info


@router.delete("/{car_id}", summary="Remove a car from the current user")
async def delete_car(
    car_id: Annotated[str, Path(max_length=MAX_CAR_ID_CHARS, pattern=NO_NUL)],
    current_user: dict = Depends(get_current_user),
):
    result = await user_profile_service.remove_car_from_user(current_user["id"], car_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail="Car not found")
    return {"deleted": car_id}
