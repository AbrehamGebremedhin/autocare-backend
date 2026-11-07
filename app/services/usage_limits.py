"""Per-account and process-wide usage ceilings for the chat route.

Everything here lives in this process's memory: the counters are per process and reset when it restarts. The
deployment runs a single worker, so that is the whole picture; with more workers each would allow its own share.
"""
import math
from datetime import date, datetime, time, timedelta, timezone

from app.core.config import get_settings
from app.utils.exceptions import RateLimitExceededException

_counts: dict[tuple[str, date], int] = {}  # (user id, UTC day) -> accepted messages


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def count_daily_message(user_id: str) -> None:
    """Count one message for the account today (UTC), or raise a 429 with Retry-After when it has used up
    DAILY_MESSAGE_LIMIT (0 turns the cap off). Days before today are dropped on every call."""
    limit = get_settings().DAILY_MESSAGE_LIMIT
    if limit <= 0:
        return
    now = _utc_now()
    today = now.date()
    for key in [k for k in _counts if k[1] != today]:
        del _counts[key]
    key = (user_id, today)
    if _counts.get(key, 0) >= limit:
        midnight = datetime.combine(today + timedelta(days=1), time.min, tzinfo=timezone.utc)
        raise RateLimitExceededException(
            f"Daily message limit of {limit} reached. Try again after midnight UTC.",
            retry_after=math.ceil((midnight - now).total_seconds()),
        )
    _counts[key] = _counts.get(key, 0) + 1
