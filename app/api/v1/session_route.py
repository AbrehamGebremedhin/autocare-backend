import shutil
import tempfile
import weakref
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services import diagnosis
from app.services.diagnosis import run_turn, sse
from app.services.media import Upload, validate_uploads
from app.services.user_profile_service import UserProfileService
from app.utils.auth_middleware import get_current_user
from app.utils.exceptions import (
    BusinessLogicException,
    RecordNotFoundException,
    ResourceConflictException,
)
from app.utils.limiter import limiter
from app.utils.logger import get_logger_instance

router = APIRouter(prefix="/sessions", tags=["Sessions"])
profiles = UserProfileService()
logger = get_logger_instance("session_route")

MAX_TEXT_CHARS = 5000
BUSY = "A reply is still being generated for this session."
DELETING = "This session is being deleted."

# ponytail: in-process, per-instance. Move to a DB advisory lock if we ever run >1 instance.
_in_flight: set[str] = set()  # sessions with a turn running
_deleting: set[str] = set()  # sessions a DELETE is removing; no turn may start (and upload media)


class SessionCreate(BaseModel):
    car_id: str


async def _owned_session(session_id: str, user_id: str) -> dict:
    session = await diagnosis.sessions.get_for_user(session_id, user_id)
    if session is None:
        raise RecordNotFoundException("Session")
    return session


def _release(session_id: str, tmpdir: str | None) -> None:
    """End of a turn: free the session's in-flight lock and delete the turn's upload dir."""
    _in_flight.discard(session_id)
    if tmpdir is not None:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _events(
    session: dict, car: dict, text: str | None, uploads: Sequence[Upload] = (), tmpdir: str | None = None
) -> tuple[AsyncIterator[str], weakref.finalize]:
    """SSE body for one turn, plus the release for the in-flight lock the caller already holds.

    Release frees the lock and removes `tmpdir` (where `uploads` live, so it must outlive the
    response). The body releases when it completes, fails or is closed by a client disconnect;
    the finalizer also releases if the body is dropped without ever being iterated. Calling
    `release` runs it once and detaches it, so a stale body can never free a later turn's lock.
    """

    async def stream():
        try:
            async with aclosing(run_turn(session, car, text, list(uploads))) as events:
                async for event, data in events:
                    yield sse(event, data)
        except Exception as exc:
            await logger.error(f"turn failed: {type(exc).__name__}")
            yield sse("error", {"message": diagnosis.LLM_ERROR})
        finally:
            release()

    body = stream()
    release = weakref.finalize(body, _release, session["id"], tmpdir)
    return body, release


@router.post("", summary="Start a diagnosis session for one of the user's cars")
async def create_session(payload: SessionCreate, current_user: dict = Depends(get_current_user)):
    cars = await profiles.get_user_cars(current_user["id"])
    if not any(c.get("id") == payload.car_id for c in cars):
        raise RecordNotFoundException("Car")
    return await diagnosis.sessions.create({"user_id": current_user["id"], "car_id": payload.car_id})


@router.get("", summary="List the user's sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    return await diagnosis.sessions.list_for_user(current_user["id"])


@router.get("/{session_id}", summary="Get one session with its messages and state")
async def get_session(session_id: str, current_user: dict = Depends(get_current_user)):
    return await _owned_session(session_id, current_user["id"])


@router.delete("/{session_id}", summary="Delete a session and its media")
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    session = await _owned_session(session_id, current_user["id"])
    sid = session["id"]
    if sid in _in_flight:
        raise ResourceConflictException(BUSY)
    if sid in _deleting:
        raise ResourceConflictException(DELETING)
    _deleting.add(sid)
    try:
        await diagnosis.sessions.delete({"id": sid, "user_id": current_user["id"]})
        await diagnosis.media_store.delete_prefix(session["user_id"], sid)  # best effort, never raises
    finally:
        _deleting.discard(sid)
    return {"deleted": sid}


@router.post("/{session_id}/messages", summary="Send a message; the reply streams back as SSE")
@limiter.limit("20/minute")
async def send_message(
    request: Request,
    session_id: str,
    text: str | None = Form(None, max_length=MAX_TEXT_CHARS),
    files: list[UploadFile] = File([]),
    current_user: dict = Depends(get_current_user),
):
    text = (text or "").replace(chr(0), "").strip() or None
    if not text and not files:
        raise BusinessLogicException("A message needs text or at least one file.")
    session = await _owned_session(session_id, current_user["id"])
    car = await profiles.car_crud.get_car_by_id(session["car_id"])
    if car is None:
        raise RecordNotFoundException("Car")

    sid = session["id"]
    if sid in _deleting:
        raise ResourceConflictException(DELETING)
    if sid in _in_flight:
        raise ResourceConflictException(BUSY)
    _in_flight.add(sid)
    # From here every exit releases: the stream's own release once it exists, else _release.
    # Uploads are validated before the response starts, so their errors are plain JSON 4xx.
    tmpdir: str | None = None
    release = None
    try:
        # Re-read under the lock: the snapshot above may predate the previous turn's writes.
        session = await _owned_session(sid, current_user["id"])
        uploads: list[Upload] = []
        if files:
            tmpdir = tempfile.mkdtemp(prefix="autocare-turn-")
            uploads = await validate_uploads(files, Path(tmpdir))
        body, release = _events(session, car, text, uploads, tmpdir)
        return StreamingResponse(
            body,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
        )
    except BaseException:
        if release is not None:
            release()
        else:
            _release(sid, tmpdir)
        raise
