import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import datetime, timezone
from uuid import uuid4

from app.core.config import get_settings
from app.CRUD.chat_session_crud import ChatSessionCRUD
from app.db.bucket_operations import SupabaseBucketManager
from app.schemas.session import DiagnosisState, MediaRef, Message, Observations
from app.services import hearing, kb, llm, media
from app.services.media import MediaError, Upload
from app.services.prompts import build_messages, prompt_content
from app.utils.logger import get_logger_instance

logger = get_logger_instance("diagnosis")
sessions = ChatSessionCRUD()
bucket = SupabaseBucketManager()

LLM_ERROR = "The diagnosis service failed, please retry."
MEDIA_ERROR = "Could not process the attached media."

# Storage object names. Images are stored as the re-encoded JPEG the model saw; audio and video
# are the original upload bytes, whose real container is unknown (server-side names are .bin),
# so for those the extension is only a label.
_STORED_EXT = {"image": ".jpg", "video": ".mp4", "audio": ".m4a"}


class _EmptyReply(Exception):
    """The reply stream produced nothing but whitespace."""


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _message(role: str, content: str) -> dict:
    return Message(
        id=uuid4().hex, role=role, content=content, created_at=datetime.now(timezone.utc).isoformat()
    ).model_dump(mode="json")


async def _store_originals(session: dict, message_id: str, uploads: list[Upload]) -> list[dict]:
    """Upload each original to Storage. Never raises: a failed file is just left out of the refs."""
    bucket_name = get_settings().MEDIA_BUCKET

    async def one(n: int, u: Upload) -> dict | None:
        path = f"{session['user_id']}/{session['id']}/{message_id}/{n}{_STORED_EXT[u.kind]}"
        content_type = "image/jpeg" if u.kind == "image" else "application/octet-stream"
        try:
            await bucket.upload_file(bucket_name, str(u.path), path, content_type)
        except Exception as exc:
            await logger.warning(f"media upload to storage failed: {type(exc).__name__}")
            return None
        return MediaRef(kind=u.kind, path=path).model_dump()

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


async def run_turn(
    session: dict, car: dict, text: str | None, uploads: list[Upload]
) -> AsyncIterator[tuple[str, dict]]:
    settings = get_settings()
    previous = DiagnosisState.model_validate(session.get("state") or {})
    history = session.get("messages") or []

    # Saved before the first event, so a client that goes away early never loses its message.
    user = _message("user", text or "")
    await sessions.append_messages(session["id"], [user])

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
        await sessions.update_message(session["id"], user["id"], patch)
        if failed:
            while isinstance(seen, BaseExceptionGroup):  # TaskGroups nest; log the first real cause
                seen = seen.exceptions[0]
            detail = f": {seen.detail[:300]}" if isinstance(seen, MediaError) and seen.detail else ""
            await logger.error(f"media processing failed: {type(seen).__name__}{detail}")
            yield "error", {"message": MEDIA_ERROR}
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

    prompt = build_messages(
        car=car,
        state=previous,
        history=history,
        text=text,
        observations=observations,
        passages=passages,
        images=images,
    )
    parts: list[str] = []
    try:
        async with aclosing(llm.stream_reply(prompt)) as stream:
            async for token in stream:
                parts.append(token)
                yield "token", {"text": token}
        reply = "".join(parts)
        if not reply.strip():
            raise _EmptyReply
        n = settings.HISTORY_MESSAGES
        recent = [{"role": m["role"], "content": prompt_content(m["content"])} for m in (history[-n:] if n > 0 else [])]
        state = await llm.extract_state(
            [*recent, {"role": "user", "content": prompt_content(text)}, {"role": "assistant", "content": reply}],
            previous,
        )
    except Exception as exc:
        await logger.error(f"diagnosis LLM call failed: {type(exc).__name__}")
        yield "error", {"message": LLM_ERROR}
        return

    assistant = _message("assistant", reply)
    await sessions.append_messages(session["id"], [assistant])
    await sessions.save_state(session["id"], state.model_dump(), state.title or session.get("title", ""))
    yield "state", state.model_dump()
    yield "done", {"message_id": assistant["id"]}
