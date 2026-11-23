from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response
from pydantic import BaseModel, Field

from app.schemas.Car import MAX_CAR_ID_CHARS, NO_NUL
from app.services import videos
from app.services.user_profile_service import UserProfileService
from app.utils.auth_middleware import get_current_user
from app.utils.exceptions import RecordNotFoundException

router = APIRouter(prefix="/cars/{car_id}/videos", tags=["Videos"])
profiles = UserProfileService()

CarId = Annotated[str, Path(max_length=MAX_CAR_ID_CHARS, pattern=NO_NUL)]


class NotHelpful(BaseModel):
    job_key: str = Field(pattern=r"^[a-z0-9-]{1,60}$")
    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")


@router.post("/not-helpful", status_code=204, summary="Hide a suggested video for this user, car and job")
async def not_helpful(car_id: CarId, body: NotHelpful, user: dict = Depends(get_current_user)):
    if car_id not in await profiles.garage_ids(user["id"]):
        raise RecordNotFoundException("Car")  # not in this garage: 404, never reveal whether it exists
    await videos.store.add_feedback(user["id"], car_id, body.job_key, body.video_id)
    return Response(status_code=204)
