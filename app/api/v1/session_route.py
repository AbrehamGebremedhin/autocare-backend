import weakref
from collections.abc import AsyncIterator
from contextlib import aclosing

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services import diagnosis
from app.services.diagnosis import run_turn, sse
from app.services.user_profile_service import UserProfileService
from app.utils.auth_middleware import get_current_user
from app.utils.exceptions import (
    BaseAPIException,
    BusinessLogicException,
    ErrorCode,
    RecordNotFoundException,
    ResourceConflictException,
)
from app.utils.limiter import limiter
from app.utils.logger import get_logger_instance

router = APIRouter(prefix="/sessions", tags=["Sessions"])
profiles = UserProfileService()
logger = get_logger_instance("session_route")

MAX_TEXT_CHARS = 5000

# ponytail: in-process, per-instance. Move to a DB advisory lock if we ever run >1 instance.
_in_flight: set[str] = set()


class SessionCreate(BaseModel):
    car_id: str


async def _owned_session(session_id: str, user_id: str) -> dict:
    session = await diagnosis.sessions.get_for_user(session_id, user_id)
    if session is None:
        raise RecordNotFoundException("Session")
    return session


def _events(session: dict, car: dict, text: str | None) -> tuple[AsyncIterator[str], weakref.finalize]:
    """SSE body for one turn, plus the release for the in-flight lock the caller already holds.

    The body releases the lock when it completes, fails or is closed by a client disconnect;
    the finalizer also releases it if the body is dropped without ever being iterated. Calling
    `release` runs it once and detaches it, so a stale body can never free a later turn's lock.
    """

    async def stream():
        try:
            async with aclosing(run_turn(session, car, text, [])) as events:
                async for event, data in events:
                    yield sse(event, data)
        except Exception as exc:
            await logger.error(f"turn failed: {type(exc).__name__}")
            yield sse("error", {"message": diagnosis.LLM_ERROR})
        finally:
            release()

    body = stream()
    release = weakref.finalize(body, _in_flight.discard, session["id"])
    return body, release


@router.post("", summary="Start a diagnosis session for one of the user's cars")
async def create_session(payload: SessionCreate, current_user: dict = Depends(get_current_user)):
    cars = await profiles.get_user_cars(current_user["id"])
    if not any(c.get("id") == payload.car_id for c in cars):
        raise RecordNotFoundException("Car")
    rows = await diagnosis.sessions.create({"user_id": current_user["id"], "car_id": payload.car_id})
    return rows[0]


@router.get("", summary="List the user's sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    return await diagnosis.sessions.list_for_user(current_user["id"])


@router.get("/{session_id}", summary="Get one session with its messages and state")
async def get_session(session_id: str, current_user: dict = Depends(get_current_user)):
    return await _owned_session(session_id, current_user["id"])


@router.delete("/{session_id}", summary="Delete a session")
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    session = await _owned_session(session_id, current_user["id"])
    await diagnosis.sessions.delete({"id": session["id"], "user_id": current_user["id"]})
    return {"deleted": session["id"]}


@router.post("/{session_id}/messages", summary="Send a message; the reply streams back as SSE")
@limiter.limit("20/minute")
async def send_message(
    request: Request,
    session_id: str,
    text: str | None = Form(None, max_length=MAX_TEXT_CHARS),
    files: list[UploadFile] = File([]),
    current_user: dict = Depends(get_current_user),
):
    text = (text or "").strip() or None
    if not text and not files:
        raise BusinessLogicException("A message needs text or at least one file.")
    if files:  # replaced in Task 14
        raise BaseAPIException(
            ErrorCode.INVALID_FORMAT, "Media upload is not enabled yet.", status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        )
    session = await _owned_session(session_id, current_user["id"])
    car = await profiles.car_crud.get_car_by_id(session["car_id"])
    if car is None:
        raise RecordNotFoundException("Car")

    sid = session["id"]
    if sid in _in_flight:
        raise ResourceConflictException("A reply is still being generated for this session.")
    _in_flight.add(sid)
    body, release = _events(session, car, text)
    try:
        return StreamingResponse(
            body, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
    except BaseException:
        release()
        raise
