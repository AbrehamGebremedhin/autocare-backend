import asyncio
import json
import re
from functools import cache
from typing import AsyncIterator

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.schemas.session import DiagnosisState
from app.services.prompts import STATE_PROMPT
from app.utils.logger import get_logger_instance

logger = get_logger_instance("llm")

_STATE_HISTORY = 10
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


@cache
def get_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(
        base_url=s.DEEPSEEK_BASE_URL,
        api_key=s.DEEPSEEK_API_KEY,
        timeout=120,
        max_retries=1,
    )


async def stream_reply(messages: list[dict]) -> AsyncIterator[str]:
    stream = await get_client().chat.completions.create(
        model=get_settings().DEEPSEEK_MODEL, messages=messages, stream=True
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        text = getattr(chunk.choices[0].delta, "content", None)
        if text:
            yield text


def _first_json_object(text: str) -> dict:
    text = _FENCE.sub("", text)
    start = text.index("{")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    return obj


async def extract_state(conversation: list[dict], previous: DiagnosisState) -> DiagnosisState:
    messages = [
        {"role": "system", "content": STATE_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "previous_state": previous.model_dump(),
                    "conversation": conversation[-_STATE_HISTORY:],
                },
                ensure_ascii=False,
            ),
        },
    ]
    try:
        resp = await get_client().chat.completions.create(
            model=get_settings().DEEPSEEK_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return DiagnosisState.model_validate(_first_json_object(resp.choices[0].message.content or ""))
    except Exception as exc:
        await logger.warning(f"extract_state failed, keeping previous state: {type(exc).__name__}")
        return previous


async def ping() -> bool:
    try:
        await asyncio.wait_for(get_client().models.list(), 5)
        return True
    except Exception:
        return False
