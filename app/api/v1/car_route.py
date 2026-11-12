from datetime import date
from typing import Annotated, List, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Path, Request
from pydantic import BaseModel, Field, field_validator

from app.CRUD.car_crud import ingest_car_manual
from app.core.config import get_settings
from app.schemas.Car import MAX_CAR_ID_CHARS, MAX_CAR_NAME_CHARS, MIN_CAR_YEAR, NO_NUL, CarBase
from app.services import usage_limits
from app.services.user_profile_service import UserProfileService
from app.utils.auth_middleware import get_current_user
from app.utils.exceptions import BusinessLogicException, RecordNotFoundException, ValidationException
from app.utils.limiter import limiter

router = APIRouter(prefix="/cars", tags=["Cars"])

user_profile_service = UserProfileService()


class CarCreate(BaseModel):
    make: str = Field(min_length=1, max_length=MAX_CAR_NAME_CHARS, pattern=NO_NUL)
    model: str = Field(min_length=1, max_length=MAX_CAR_NAME_CHARS, pattern=NO_NUL)
    year: int = Field(ge=MIN_CAR_YEAR)
    transmission: Literal["manual", "automatic", "cvt", "dual_clutch"] | None = None

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
@limiter.limit("10/minute")
async def add_car(
    request: Request,
    car: CarCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    # Every new car starts paid background work (LLM, searches, downloads), so the cost per account is bounded here.
    cap = get_settings().MAX_CARS_PER_USER
    if cap > 0:
        garage = await user_profile_service.garage_ids(current_user["id"])
        if len(garage) >= cap and user_profile_service.car_crud.unique_logic(car.make, car.model, car.year) not in garage:
            raise BusinessLogicException(f"A garage holds at most {cap} cars. Remove one to add another.")
    usage_limits.count_daily_car_add(current_user["id"])
    result = await user_profile_service.add_car_to_user(
        user_id=current_user["id"], make=car.make, model=car.model, year=car.year, transmission=car.transmission
    )
    if not result["success"]:
        raise ValidationException("Could not add car")
    car_info = result["car_info"]
    if car_info.get("kb_status", "pending") == "pending":
        background_tasks.add_task(ingest_car_manual, car_info)
    return car_info


class CarUpdate(BaseModel):
    transmission: Literal["manual", "automatic", "cvt", "dual_clutch"]


@router.patch("/{car_id}", summary="Set the transmission of one of the current user's cars")
async def update_car(
    car_id: Annotated[str, Path(max_length=MAX_CAR_ID_CHARS, pattern=NO_NUL)],
    update: CarUpdate,
    current_user: dict = Depends(get_current_user),
):
    if not await user_profile_service.set_transmission(current_user["id"], car_id, update.transmission):
        raise RecordNotFoundException("Car")  # not in this garage: 404, never reveal whether it exists
    return {"id": car_id, "transmission": update.transmission}


@router.delete("/{car_id}", summary="Remove a car from the current user")
async def delete_car(
    car_id: Annotated[str, Path(max_length=MAX_CAR_ID_CHARS, pattern=NO_NUL)],
    current_user: dict = Depends(get_current_user),
):
    result = await user_profile_service.remove_car_from_user(current_user["id"], car_id)
    if not result["success"]:
        raise RecordNotFoundException("Car")
    return {"deleted": car_id}
