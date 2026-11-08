import asyncio
import hashlib
import re
import time
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.CRUD import vehicle_visual_crud as crud
from app.core.config import get_settings
from app.schemas.Car import MAX_CAR_ID_CHARS, NO_NUL
from app.services import vehicle_visuals
from app.services.visual_signing import verify
from app.services.visual_store import VisualStore
from app.utils.auth_middleware import get_current_user
from app.utils.exceptions import RateLimitExceededException, RecordNotFoundException
from app.utils.limiter import limiter
from app.utils.logger import get_logger_instance

cars_router = APIRouter(prefix="/cars", tags=["Vehicle visuals"])
files_router = APIRouter(prefix="/visuals", tags=["Vehicle visuals"])
logger = get_logger_instance("visual_route")

MAX_ARTWORK_BYTES = 409_600
_NAME = re.compile(r"[0-9a-f]{64}\.webp")

CarId = Annotated[str, Path(max_length=MAX_CAR_ID_CHARS, pattern=NO_NUL)]


class PrepareBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant_key: str | None = Field(default=None, min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")


async def _car_of(user: dict, car_id: str) -> dict:
    car = await crud.owned_car(user["id"], car_id)
    if car is None:
        raise RecordNotFoundException("Car")
    return car


@cars_router.get("/{car_id}/visual", summary="Artwork descriptor for one of the current user's cars")
async def get_visual(car_id: CarId, current_user: dict = Depends(get_current_user)):
    car = await _car_of(current_user, car_id)
    try:
        descriptor = await vehicle_visuals.get_descriptor(current_user["id"], car)
    except vehicle_visuals.CarNotOwned:
        raise RecordNotFoundException("Car") from None
    return descriptor.to_dict()


@cars_router.post("/{car_id}/visual/prepare", summary="Reuse or queue artwork for one of the current user's cars")
async def prepare_visual(
    car_id: CarId,
    body: Annotated[PrepareBody | None, Body()] = None,
    current_user: dict = Depends(get_current_user),
):
    car = await _car_of(current_user, car_id)
    try:
        descriptor, created = await vehicle_visuals.prepare(
            current_user["id"], car, body.variant_key if body else None
        )
    except vehicle_visuals.CarNotOwned:
        raise RecordNotFoundException("Car") from None
    except vehicle_visuals.VariantMismatch:
        # Same 422 body as any other invalid field: names only, never the submitted value.
        raise RequestValidationError([{"loc": ("body", "variant_key"), "type": "value_error", "msg": "invalid"}]) from None
    except vehicle_visuals.Throttled as exc:
        raise RateLimitExceededException(retry_after=exc.retry_after) from None
    return JSONResponse(descriptor.to_dict(), status_code=202 if created else 200)


def _not_found() -> RecordNotFoundException:
    # One message for unknown, expired, tampered and missing: the response never says which.
    return RecordNotFoundException("File")


def _read_artwork(path) -> bytes | None:
    try:
        with open(path, "rb") as f:
            data = f.read(MAX_ARTWORK_BYTES + 1)
    except OSError:
        return None
    return data if len(data) <= MAX_ARTWORK_BYTES else None


@files_router.get("/files/{name:path}", summary="Signed artwork download (no bearer token; the url is the credential)")
@limiter.limit("120/minute")
async def download_visual(
    request: Request, name: str, exp: str | None = None, sig: str | None = None
):
    if not _NAME.fullmatch(name):
        raise _not_found()
    sha = name[:-5]
    if not verify(sha, exp, sig):
        raise _not_found()
    path = VisualStore().object_path(sha)
    if path is None:
        raise _not_found()
    data = await asyncio.to_thread(_read_artwork, path)
    if data is None or hashlib.sha256(data).hexdigest() != sha:
        await logger.warning("visual download refused: stored object is unreadable, oversized or corrupt")
        raise _not_found()
    ttl = get_settings().VISUAL_URL_TTL_SECONDS
    max_age = max(0, min(ttl, int(exp) - int(time.time())))
    return Response(
        content=data,
        media_type="image/webp",
        headers={
            "Content-Length": str(len(data)),
            "Cache-Control": f"private, max-age={max_age}",
            "X-Content-Type-Options": "nosniff",
        },
    )
