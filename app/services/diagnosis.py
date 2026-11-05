import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import datetime, timezone
from uuid import uuid4

from app.core.config import get_settings
from app.CRUD.chat_session_crud import ChatSessionCRUD
from app.schemas.session import DiagnosisState, Message
from app.services import kb, llm
from app.services.media import Upload
from app.services.prompts import build_messages
from app.utils.logger import get_logger_instance

logger = get_logger_instance("diagnosis")
sessions = ChatSessionCRUD()

LLM_ERROR = "The diagnosis service failed, please retry."


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _message(role: str, content: str) -> dict:
    return Message(
        id=uuid4().hex, role=role, content=content, created_at=datetime.now(timezone.utc).isoformat()
    ).model_dump(mode="json")


async def run_turn(
    session: dict, car: dict, text: str | None, uploads: list[Upload]
) -> AsyncIterator[tuple[str, dict]]:
    settings = get_settings()
    previous = DiagnosisState.model_validate(session.get("state") or {})
    history = session.get("messages") or []

    yield "stage", {"name": "searching"}
    await sessions.append_messages(session["id"], [_message("user", text or "")])

    passages = []
    if text:
        try:
            passages = await kb.search(text, car["id"], settings.KB_TOP_K)
        except Exception as exc:
            await logger.warning(f"kb.search failed, replying without passages: {type(exc).__name__}")

    prompt = build_messages(
        car=car, state=previous, history=history, text=text, observations=None, passages=passages, images=[]
    )
    parts: list[str] = []
    try:
        async with aclosing(llm.stream_reply(prompt)) as stream:
            async for token in stream:
                parts.append(token)
                yield "token", {"text": token}
        reply = "".join(parts)
        n = settings.HISTORY_MESSAGES
        recent = [{"role": m["role"], "content": m["content"]} for m in (history[-n:] if n > 0 else [])]
        state = await llm.extract_state(
            [*recent, {"role": "user", "content": text or ""}, {"role": "assistant", "content": reply}], previous
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
