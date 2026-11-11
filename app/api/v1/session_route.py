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
from app.schemas.session import DiagnosisState
from app.services import diagnosis
from app.services.diagnosis import Lease, run_turn, sse
from app.services.media import Upload, validate_uploads
from app.services.usage_limits import count_daily_message, free_media_slot, take_media_slot
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


async def _claim(session_id: str, user_id: str) -> Lease:
    """The session's lease, else 409; 404 when the session was deleted since the caller's ownership check."""
    holder = await diagnosis.sessions.claim(session_id, user_id, diagnosis.LEASE_SECONDS)
    if holder is None:
        await _owned_session(session_id, user_id)  # gone meanwhile: 404, not "busy"
        raise ResourceConflictException(BUSY)
    return Lease(session_id, holder)


async def _free_lease(lease: Lease) -> None:
    try:
        await diagnosis.sessions.release(lease.session_id, lease.holder)
    except Exception as exc:
        await logger.warning(f"session lease release failed, it will expire: {type(exc).__name__}")


def _spawn_release(lease: Lease, media_user: str | None = None) -> asyncio.Task:
    if media_user is not None:
        free_media_slot(media_user)  # here, on the loop: the slot set is not thread-safe
    task = asyncio.get_running_loop().create_task(_free_lease(lease))
    _releasing.add(task)
    task.add_done_callback(_releasing.discard)
    return task


def _release(
    loop: asyncio.AbstractEventLoop, lease: Lease, tmpdir: str | None, media_user: str | None = None
) -> asyncio.Task | None:
    """End of a turn: delete the turn's upload dir, give back its account's media slot (if it holds one) and free the
    session's lease.

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
        return _spawn_release(lease, media_user)
    try:
        loop.call_soon_threadsafe(_spawn_release, lease, media_user)
    except RuntimeError:
        pass  # the loop is closed
    return None


def _events(
    lease: Lease,
    session: dict,
    car: dict,
    text: str | None,
    uploads: Sequence[Upload] = (),
    tmpdir: str | None = None,
    media_user: str | None = None,
    retry_of: str | None = None,
) -> tuple[AsyncIterator[str], weakref.finalize]:
    """SSE body for one turn, plus the release for the session lease (and, if `media_user`, that account's media slot) the
    caller already holds.

    Release frees the lease and the slot and removes `tmpdir` (where `uploads` live, so it must outlive the
    response). The body releases when it completes, fails or is closed by a client disconnect;
    the finalizer also releases if the body is dropped without ever being iterated. Calling
    `release` runs it once and detaches it, so a stale body can never free a later turn's lease.
    Must be called on the event loop that serves the response.
    """

    async def finish() -> None:
        # Before the turn's last event (its writes are done): a client that sends its next message as soon
        # as it sees `done` must find the lease free. Shielded: being cancelled here must not cancel it.
        pending = release()
        if pending is not None:
            await asyncio.shield(pending)

    async def stream():
        try:
            async with aclosing(run_turn(session, car, text, list(uploads), lease, retry_of)) as events:
                async for event, data in events:
                    if event in ("done", "error"):
                        await finish()
                    yield sse(event, data)
        except Exception as exc:
            await logger.error(f"turn failed: {type(exc).__name__}")
            await finish()
            yield sse("error", {"message": diagnosis.LLM_ERROR})
        finally:
            release()  # backstop (disconnect, cancellation): starts the release, never awaits it

    body = stream()
    release = weakref.finalize(body, _release, asyncio.get_running_loop(), lease, tmpdir, media_user)
    return body, release


@router.post("", summary="Start a diagnosis session for one of the user's cars")
@limiter.limit("30/minute")
async def create_session(request: Request, payload: SessionCreate, current_user: dict = Depends(get_current_user)):
    cars = await profiles.get_user_cars(current_user["id"])
    if not any(c.get("id") == payload.car_id for c in cars):
        raise RecordNotFoundException("Car")
    return await diagnosis.sessions.create({"user_id": current_user["id"], "car_id": payload.car_id})


@router.get("", summary="List the user's sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    return await diagnosis.sessions.list_for_user(current_user["id"])


@router.get("/{session_id}", summary="Get one session with its messages and state")
async def get_session(session_id: str, current_user: dict = Depends(get_current_user)):
    session = await _owned_session(session_id, current_user["id"])
    # Sessions saved before ranks and severities are upgraded on the way out (no float likelihoods leave the API).
    return {**session, "state": DiagnosisState.model_validate(session.get("state") or {}).model_dump()}


@router.delete("/{session_id}", summary="Delete a session and its media")
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    session = await _owned_session(session_id, current_user["id"])
    sid = session["id"]
    # The lease keeps a turn from starting while the row goes; after that no claim can succeed, so no turn
    # can store media while the folder is removed.
    lease = await _claim(sid, current_user["id"])
    try:
        await diagnosis.sessions.delete({"id": sid, "user_id": current_user["id"]})  # the lease goes with the row
    except BaseException:
        _release(asyncio.get_running_loop(), lease, None)
        raise
    await diagnosis.media_store.delete_prefix(session["user_id"], sid)  # best effort, never raises
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
    return await _start_turn(session_id, text, files, current_user)


@router.post(
    "/{session_id}/messages/{message_id}/retry",
    summary="Answer a user message again after its reply failed; nothing new is added to the transcript",
)
@limiter.limit("20/minute")
async def retry_message(
    request: Request, session_id: str, message_id: str, current_user: dict = Depends(get_current_user)
):
    return await _start_turn(session_id, None, [], current_user, retry_of=message_id)


async def _start_turn(
    session_id: str, text: str | None, files: list[UploadFile], current_user: dict, retry_of: str | None = None
) -> StreamingResponse:
    session = await _owned_session(session_id, current_user["id"])
    car = await profiles.car_crud.get_car_by_id(session["car_id"])
    if car is None:
        raise RecordNotFoundException("Car")
    car = {**car, "transmission": await profiles.get_transmission(current_user["id"], session["car_id"])}

    # A media turn takes its account's slot before anything is claimed or stored (429 if it is taken).
    media_user = current_user["id"] if files else None
    if media_user is not None:
        take_media_slot(media_user)
    sid = session["id"]
    try:
        lease = await _claim(sid, current_user["id"])
    except BaseException:
        if media_user is not None:
            free_media_slot(media_user)
        raise
    # From here every exit releases: the stream's own release once it exists, else _release.
    # Uploads are validated before the response starts, so their errors are plain JSON 4xx.
    tmpdir: str | None = None
    release = None
    try:
        # Re-read under the lease: the snapshot above may predate the previous turn's writes.
        session = await _owned_session(sid, current_user["id"])
        if retry_of is not None:
            text = diagnosis.retry_target(session.get("messages") or [], retry_of)["content"] or None
        uploads: list[Upload] = []
        if files:
            tmpdir = tempfile.mkdtemp(prefix="autocare-turn-")
            uploads = await validate_uploads(files, Path(tmpdir))
        # Last check before the model: only a request that got this far uses up the day's count. A 429 here
        # releases the lease and the slot like any other error.
        count_daily_message(current_user["id"])
        body, release = _events(lease, session, car, text, uploads, tmpdir, media_user, retry_of)
        return StreamingResponse(
            body,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
        )
    except BaseException as exc:
        pending = release() if release is not None else _release(asyncio.get_running_loop(), lease, tmpdir, media_user)
        if pending is not None and isinstance(exc, Exception):
            await asyncio.shield(pending)  # a plain error (say a 415): answer once the lease is free
        raise
