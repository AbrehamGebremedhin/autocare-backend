import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from app.core.config import get_settings
from app.CRUD.chat_session_crud import ChatSessionCRUD
from app.schemas.session import DiagnosisState, MediaRef, Message, Observations, TestDone, merge_state
from app.services import findings, hearing, kb, llm, media
from app.services.media import MediaError, Upload
from app.services.media_store import MediaStore
from app.services.prompts import build_messages, clean_citations, history_turn, prompt_content
from app.utils.exceptions import BusinessLogicException, RecordNotFoundException
from app.utils.logger import get_logger_instance

logger = get_logger_instance("diagnosis")
sessions = ChatSessionCRUD()
media_store = MediaStore()

LLM_ERROR = "The diagnosis service failed, please retry."
LEASE_LOST = "This reply was stopped because the session changed meanwhile, please retry."
LEASE_SECONDS = 300.0  # how long a claimed or renewed session lease lasts without a renewal
RENEW_EVERY_SECONDS = 60.0  # the heartbeat period, well inside LEASE_SECONDS
MEDIA_ERROR = "Could not process the attached media."

# Stored file names. Images are stored as the re-encoded JPEG the model saw; audio and video
# are the original upload bytes, whose real container is unknown (server-side names are .bin),
# so for those the extension is only a label.
_STORED_EXT = {"image": ".jpg", "video": ".mp4", "audio": ".m4a"}


class _EmptyReply(Exception):
    """The reply stream produced nothing but whitespace."""


class _LeaseLost(Exception):
    """The turn's session lease is held by someone else now (or the session is gone)."""


@dataclass
class Lease:
    """The session lease a turn holds (ChatSessionCRUD.claim): run_turn keeps it alive and marks it lost."""

    session_id: str
    holder: str
    lost: bool = False


def _lost(lease: Lease | None) -> bool:
    return lease is not None and lease.lost


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _message(role: str, content: str, **extra) -> dict:
    return Message(
        id=uuid4().hex, role=role, content=content, created_at=datetime.now(timezone.utc).isoformat(), **extra
    ).model_dump(mode="json")


def _failed(m: dict) -> bool:
    return m.get("status") == "failed"


def retry_target(messages: list[dict], message_id: str) -> dict:
    """The user message a retry answers again: it must be the last user message, with nothing after it but
    failed attempts, and carry no files (those are not kept for a second try; the user sends them again)."""
    idx = next((i for i, m in enumerate(messages) if m["id"] == message_id and m["role"] == "user"), None)
    if idx is None:
        raise RecordNotFoundException("Message")
    if not all(_failed(m) for m in messages[idx + 1:]) or len(messages) == idx + 1:
        raise BusinessLogicException("Only a message whose reply failed can be retried.")
    if messages[idx].get("media"):
        raise BusinessLogicException("This message had files attached; send it again with the files.")
    return messages[idx]


async def _store_originals(session: dict, message_id: str, uploads: list[Upload]) -> list[dict]:
    """Copy each original into the media store. Never raises: a failed file is just left out of the refs."""

    async def one(n: int, u: Upload) -> dict | None:
        try:
            ref = await media_store.upload(
                session["user_id"], session["id"], message_id, n, u.kind, _STORED_EXT[u.kind], u.path
            )
        except Exception as exc:
            await logger.warning(f"media store write failed: {type(exc).__name__}")
            return None
        return MediaRef(**ref).model_dump()

    refs = await asyncio.gather(*(one(n, u) for n, u in enumerate(uploads)))
    return [r for r in refs if r is not None]


# TaskGroups below, not gather: when one job fails its siblings are cancelled (killing their
# ffmpeg processes) and awaited, so none of a turn's work outlives the turn.


async def _sounds(u: Upload) -> list:
    pcm = await media.audio_pcm(u.path, sr=hearing.CLAP_SR)
    return await hearing.tag_sounds(pcm, sr=hearing.CLAP_SR)


async def _hear(u: Upload) -> Observations:
    # transcribe() decodes its own 16 kHz copy, so both decodes run side by side.
    async with asyncio.TaskGroup() as tg:
        tags = tg.create_task(_sounds(u))
        transcript = tg.create_task(hearing.transcribe(u.path))
    return Observations(transcript=transcript.result(), sounds=tags.result())


async def _look_and_listen(u: Upload, max_frames: int) -> tuple[list[bytes], Observations | None]:
    if u.kind == "image":
        return [await asyncio.to_thread(u.path.read_bytes)], None
    async with asyncio.TaskGroup() as tg:
        frames = tg.create_task(media.extract_frames(u.path, u.duration or 0.0, max_frames)) if u.kind == "video" else None
        heard = tg.create_task(_hear(u)) if u.kind == "audio" or u.has_audio else None
    images = frames.result() if frames else []
    if not images and heard is None:  # silent video whose frames all failed to decode
        raise MediaError(detail="no decodable frames and no audio")
    return images, heard.result() if heard else None


async def _observe(uploads: list[Upload]) -> tuple[list[bytes], Observations | None]:
    """JPEGs for the model (photos or video frames) and what was heard (None if nothing had audio)."""
    max_frames = get_settings().MAX_FRAMES
    async with asyncio.TaskGroup() as tg:
        jobs = [tg.create_task(_look_and_listen(u, max_frames)) for u in uploads]
    results = [j.result() for j in jobs]
    images = [img for imgs, _ in results for img in imgs]
    heard = [o for _, o in results if o is not None]
    if not heard:
        return images, None
    transcript = " ".join(o.transcript for o in heard if o.transcript)
    return images, Observations(transcript=transcript, sounds=[s for o in heard for s in o.sounds])


async def _heartbeat(lease: Lease) -> None:
    """Renew the lease every RENEW_EVERY_SECONDS, however long a stage takes (a model call can wait minutes).
    A renew that fails (database hiccup) is retried at the next beat, until the last good renewal is so old
    that the lease may run out before the next beat: then it counts as lost. One that finds the lease held by
    someone else, or the session gone, marks it lost too. Either way the heartbeat stops."""
    last_renewed = time.monotonic()
    while True:
        await asyncio.sleep(RENEW_EVERY_SECONDS)
        try:
            held = await sessions.renew(lease.session_id, lease.holder, LEASE_SECONDS)
        except Exception as exc:
            await logger.warning(f"session lease renew failed: {type(exc).__name__}")
            if time.monotonic() - last_renewed < LEASE_SECONDS - RENEW_EVERY_SECONDS:
                continue
            held = False
        if not held:
            lease.lost = True
            await logger.error("session lease lost mid-turn; the turn stops without saving more")
            return
        last_renewed = time.monotonic()


async def run_turn(
    session: dict,
    car: dict,
    text: str | None,
    uploads: list[Upload],
    lease: Lease | None = None,
    retry_of: str | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """One diagnosis turn as (event, data) pairs. With `lease` (the session lease the caller holds), a
    background task renews it while the turn runs; once it is lost the turn ends with an error event and
    writes nothing more. With `retry_of` (the id of a user message whose reply failed) that message is answered
    again: nothing is added to the transcript before the reply."""
    heartbeat = asyncio.create_task(_heartbeat(lease)) if lease is not None else None
    try:
        async with aclosing(_turn(session, car, text, uploads, lease, retry_of)) as events:
            async for event, data in events:
                if event in ("done", "error") and heartbeat is not None:
                    heartbeat.cancel()  # the turn's last event: nothing is written after it
                yield event, data
    finally:
        if heartbeat is not None:
            heartbeat.cancel()  # never awaited: this may run in a cancelled context


async def _turn(
    session: dict, car: dict, text: str | None, uploads: list[Upload], lease: Lease | None, retry_of: str | None = None
) -> AsyncIterator[tuple[str, dict]]:
    # `error` and `done` must stay terminal events: nothing may be written or yielded after either one,
    # because run_turn stops the heartbeat and the send route frees the session lease when it sees them.
    settings = get_settings()
    previous = DiagnosisState.model_validate(session.get("state") or {})
    history = session.get("messages") or []

    if retry_of is not None:  # answer the stored message again: it is not saved a second time
        user = retry_target(history, retry_of)
        text = user["content"] or None
        history = history[: history.index(user)]
    else:
        # Saved before the first event, so a client that goes away early never loses its message.
        user = _message("user", text or "")
        await sessions.append_messages(session["id"], [user])
    history = [m for m in history if not _failed(m)]  # a failed attempt has no reply to show the model

    observations: Observations | None = None
    images: list[bytes] = []
    if uploads:
        yield "stage", {"name": "processing_media"}
        # _store_originals never raises; only the processing result can be an exception.
        refs, seen = await asyncio.gather(
            _store_originals(session, user["id"], uploads), _observe(uploads), return_exceptions=True
        )
        failed = isinstance(seen, BaseException)
        if not failed:
            images, observations = seen
        patch: dict = {"media": refs}
        if observations is not None:
            patch["observations"] = observations.model_dump()
        if _lost(lease):
            yield "error", {"message": LEASE_LOST}
            return
        await sessions.update_message(session["id"], user["id"], patch)
        if failed:
            while isinstance(seen, BaseExceptionGroup):  # TaskGroups nest; log the first real cause
                seen = seen.exceptions[0]
            detail = f": {seen.detail[:300]}" if isinstance(seen, MediaError) and seen.detail else ""
            await logger.error(f"media processing failed: {type(seen).__name__}{detail}")
            yield "error", {"message": MEDIA_ERROR}
            return

    if _lost(lease):
        yield "error", {"message": LEASE_LOST}
        return
    yield "stage", {"name": "searching"}
    heard = [observations.transcript, *(s.label for s in observations.sounds)] if observations else []
    query = " ".join(filter(None, [text, *heard]))
    passages = []
    if query:
        try:
            passages = await kb.search(query, car["id"], settings.KB_TOP_K)
        except Exception as exc:
            await logger.warning(f"kb.search failed, replying without passages: {type(exc).__name__}")

    # Findings belong to a car in the owner's garage: once it is removed an old session still works, but neither
    # reads nor writes them (the owner deleted them with the car). A failed lookup also skips them.
    try:
        keep_findings = await findings.crud.in_garage(session["user_id"], car["id"])
    except Exception as exc:
        await logger.warning(f"garage lookup failed, skipping car findings this turn: {type(exc).__name__}")
        keep_findings = False
    earlier = []
    if keep_findings:
        try:
            earlier = await findings.car_history(session["user_id"], car["id"], session["id"])
        except Exception as exc:  # the history is context: a turn goes on without it
            await logger.warning(f"car history unavailable, replying without it: {type(exc).__name__}")
    prompt = build_messages(
        car=car,
        state=previous,
        history=history,
        text=text,
        observations=observations,
        passages=passages,
        images=images,
        findings=earlier,
    )
    n = settings.HISTORY_MESSAGES
    recent = [history_turn(m) for m in (history[-n:] if n > 0 else [])]
    # What the user reported is read while the reply streams; only the extraction's result is awaited afterwards.
    said = text or (observations.transcript if observations else "")
    facts_task = asyncio.create_task(llm.extract_facts(said, recent[-2:])) if said else None
    parts: list[str] = []
    try:
        async with aclosing(llm.stream_reply(prompt)) as stream:
            async for token in stream:
                if _lost(lease):
                    raise _LeaseLost
                parts.append(token)
                yield "token", {"text": token}
        reply = clean_citations("".join(parts))
        if not reply.strip():
            raise _EmptyReply
        facts = await facts_task if facts_task else []
        state = merge_state(
            previous,
            await llm.extract_state(
                [*recent, {"role": "user", "content": prompt_content(text)}, {"role": "assistant", "content": reply}],
                previous,
            ),
            [TestDone(test=f.test, result=f.result) for f in facts if f.test],
        )
        if _lost(lease):
            raise _LeaseLost
    except _LeaseLost:
        yield "error", {"message": LEASE_LOST}
        return
    except Exception as exc:
        await logger.error(f"diagnosis LLM call failed: {type(exc).__name__}")
        # Keep a record of the failed attempt, so the transcript never has two user messages in a row, and
        # tell the client which message to retry (POST .../messages/{id}/retry).
        await sessions.append_messages(session["id"], [_message("assistant", "", status="failed", retry_of=user["id"])])
        yield "error", {"message": LLM_ERROR, "retry_message_id": user["id"]}
        return
    finally:
        if facts_task is not None:
            facts_task.cancel()  # a no-op once it has finished

    assistant = _message("assistant", reply)
    await sessions.append_messages(session["id"], [assistant])
    if facts:
        await sessions.update_message(session["id"], user["id"], {"observations": (observations or Observations()).model_copy(update={"facts": facts}).model_dump()})
    await sessions.save_state(session["id"], state.model_dump(), state.title or session.get("title", ""))  # the title is stable: merge_state keeps the first
    try:  # after the state is saved, before the terminal events: a findings failure never fails the turn
        if keep_findings:
            await findings.record_turn(session["user_id"], car["id"], session["id"], facts, said, state)
    except Exception as exc:
        await logger.warning(f"findings not saved: {type(exc).__name__}")
    yield "state", state.model_dump()
    yield "done", {"message_id": assistant["id"]}
