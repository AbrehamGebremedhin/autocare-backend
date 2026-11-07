import asyncio
import shutil
import tempfile
import weakref
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.schemas.Car import MAX_CAR_ID_CHARS, NO_NUL
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
BUSY = "This session is busy: a reply is still being generated, or it is being deleted."

# A turn or a delete owns its session through the database lease (ChatSessionCRUD.claim), so the 409 holds
# across every worker. Scheduled lease releases, kept here so they are not garbage-collected mid-flight.
_releasing: set[asyncio.Task] = set()


class SessionCreate(BaseModel):
    car_id: str = Field(min_length=1, max_length=MAX_CAR_ID_CHARS, pattern=NO_NUL)


async def _owned_session(session_id: str, user_id: str) -> dict:
    session = await diagnosis.sessions.get_for_user(session_id, user_id)
    if session is None:
        raise RecordNotFoundException("Session")
    return session


async def _free_lease(session_id: str) -> None:
    try:
        await diagnosis.sessions.release(session_id)
    except Exception as exc:
        await logger.warning(f"session lease release failed, it will expire: {type(exc).__name__}")


def _spawn_release(session_id: str) -> asyncio.Task:
    task = asyncio.get_running_loop().create_task(_free_lease(session_id))
    _releasing.add(task)
    task.add_done_callback(_releasing.discard)
    return task


def _release(loop: asyncio.AbstractEventLoop, session_id: str, tmpdir: str | None) -> asyncio.Task | None:
    """End of a turn: delete the turn's upload dir and free the session's lease.

    It never awaits, because it also runs as a finalizer and inside streams that are being cancelled (where an
    await would be cancelled too): the lease is freed by a task of its own on `loop`, returned when called on
    that loop. Called from another thread (a finalizer run by a GC there) it hands the release over to the
    loop; with the loop gone the lease simply expires.
    """
    if tmpdir is not None:
        shutil.rmtree(tmpdir, ignore_errors=True)
    try:
        on_loop = asyncio.get_running_loop() is loop
    except RuntimeError:
        on_loop = False
    if on_loop:
        return _spawn_release(session_id)
    try:
        loop.call_soon_threadsafe(_spawn_release, session_id)
    except RuntimeError:
        pass  # the loop is closed
    return None


def _events(
    session: dict, car: dict, text: str | None, uploads: Sequence[Upload] = (), tmpdir: str | None = None
) -> tuple[AsyncIterator[str], weakref.finalize]:
    """SSE body for one turn, plus the release for the session lease the caller already holds.

    Release frees the lease and removes `tmpdir` (where `uploads` live, so it must outlive the
    response). The body releases when it completes, fails or is closed by a client disconnect;
    the finalizer also releases if the body is dropped without ever being iterated. Calling
    `release` runs it once and detaches it, so a stale body can never free a later turn's lease.
    Must be called on the event loop that serves the response.
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
            pending = release()
        # Reached only when the body ran to its end (not on a disconnect): the response finishes once the
        # lease is free, so the client's next message is not refused. Shielded: being cancelled here must
        # not cancel the release.
        if pending is not None:
            await asyncio.shield(pending)

    body = stream()
    release = weakref.finalize(body, _release, asyncio.get_running_loop(), session["id"], tmpdir)
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
    # The lease keeps a turn from starting (and storing media) while the media and the row go away.
    if not await diagnosis.sessions.claim(sid, current_user["id"]):
        raise ResourceConflictException(BUSY)
    try:
        await diagnosis.media_store.delete_prefix(session["user_id"], sid)  # best effort, never raises
        await diagnosis.sessions.delete({"id": sid, "user_id": current_user["id"]})  # the lease goes with the row
    except BaseException:
        _release(asyncio.get_running_loop(), sid, None)
        raise
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
    if not await diagnosis.sessions.claim(sid, current_user["id"]):
        raise ResourceConflictException(BUSY)
    # From here every exit releases: the stream's own release once it exists, else _release.
    # Uploads are validated before the response starts, so their errors are plain JSON 4xx.
    tmpdir: str | None = None
    release = None
    try:
        # Re-read under the lease: the snapshot above may predate the previous turn's writes.
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
    except BaseException as exc:
        pending = release() if release is not None else _release(asyncio.get_running_loop(), sid, tmpdir)
        if pending is not None and isinstance(exc, Exception):
            await asyncio.shield(pending)  # a plain error (say a 415): answer once the lease is free
        raise
