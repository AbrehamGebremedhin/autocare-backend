"""Turns (user, car) into the artwork descriptor the mobile client reads.

GET never creates a job; only prepare() does. Failures of the visuals tables, the store or signing degrade to a
"temporarily_unavailable" fallback so artwork can never block anything else. Throttle counters live in this
process's memory, like usage_limits.
"""
import math
from time import time

from app.CRUD import vehicle_visual_crud as crud
from app.core.config import get_settings
from app.db.pool import DatabaseUnavailable
from app.schemas.visual import Artwork, VisualDescriptor, VisualOption
from app.services import vehicle_catalog as catalog
from app.services.visual_signing import sign_url
from app.services.visual_store import VisualStore
from app.utils.logger import get_logger_instance

logger = get_logger_instance("vehicle_visuals")

_WINDOW = 3600
_GLOBAL_RETRY_AFTER = 300
_PENDING_MS = 5000
_REVIEW_MS = 30000

_prepare_hits: dict[str, list[float]] = {}  # user id -> times of prepare calls that tried to create a job


class VariantMismatch(Exception):
    """The requested variant belongs to a different make/model than the car, or is not allowlisted."""


class Throttled(Exception):
    def __init__(self, retry_after: int):
        super().__init__(f"throttled, retry after {retry_after}s")
        self.retry_after = max(1, int(retry_after))


def _now() -> float:
    return time()


def _fallback(reason: str, options: list[tuple[str, str]] | None = None) -> VisualDescriptor:
    opts = [VisualOption(variant_key=k, label=lbl) for k, lbl in options] if reason == "choose_variant" and options else None
    return VisualDescriptor(state="fallback", preset=None, reason=reason, options=opts)


def _pending(variant_key: str, ms: int = _PENDING_MS) -> VisualDescriptor:
    return VisualDescriptor(state="pending", variant_key=variant_key, style_version=crud.STYLE_VERSION, retry_after_ms=ms)


async def _from_row(variant_key: str, row: dict | None) -> VisualDescriptor:
    if row is None:
        return _pending(variant_key)
    sha = row["sha256"]
    if row["published_revision"] >= 1 and sha:
        if VisualStore().has_object(sha):
            url, expires = sign_url(sha)
            if url:
                return VisualDescriptor(
                    state="ready", variant_key=variant_key, style_version=crud.STYLE_VERSION,
                    artwork=Artwork(sha256=sha, download_url=url, expires_at=expires))
            await logger.warning(f"visual {variant_key}: could not sign a download url")
        else:
            await logger.warning(f"visual {variant_key}: published artwork is missing from the store")
        return _fallback("temporarily_unavailable")
    if row["state"] == "review":
        return _pending(variant_key, _REVIEW_MS)
    if row["state"] in ("queued", "generating"):
        return _pending(variant_key)
    return _fallback("temporarily_unavailable")


async def _unavailable(exc: Exception) -> VisualDescriptor:
    await logger.warning(f"vehicle visuals unavailable: {type(exc).__name__}")
    return _fallback("temporarily_unavailable")


def _year(car: dict) -> int | None:
    try:
        return int(car.get("year"))
    except (TypeError, ValueError):
        return None


async def _stored_choice(user_id: str, car: dict, cands: list[str]) -> str | None:
    """The stored choice if the user still owns the car and it is still a candidate; otherwise cleared and None."""
    car_id = str(car.get("id"))
    owned = await crud.owned_car(user_id, car_id)  # outside any catch: a pool outage is the route's to handle
    stored = await crud.get_choice(user_id, car_id)
    if stored is None:
        return None
    if owned is None or stored not in cands:
        await crud.clear_choice(user_id, car_id)
        return None
    return stored


async def get_descriptor(user_id: str, car: dict) -> VisualDescriptor:
    make, model, year = car.get("make"), car.get("model"), _year(car)
    try:
        res = catalog.resolve(make, model, year, await _stored_choice(user_id, car, catalog.candidates(make, model, year)))
        if res.kind != "selected":
            return _fallback("choose_variant" if res.kind == "choose" else "unknown_variant", res.options)
        return await _from_row(res.variant_key, await crud.get_visual(res.variant_key))
    except DatabaseUnavailable:
        raise
    except Exception as exc:
        return await _unavailable(exc)


async def _check_throttles(user_id: str) -> None:
    now = _now()
    for uid in [u for u, hits in _prepare_hits.items() if not [h for h in hits if h > now - _WINDOW]]:
        del _prepare_hits[uid]
    hits = [h for h in _prepare_hits.get(user_id, []) if h > now - _WINDOW]
    _prepare_hits[user_id] = hits
    s = get_settings()
    if s.VISUAL_PREPARE_PER_HOUR > 0 and len(hits) >= s.VISUAL_PREPARE_PER_HOUR:
        raise Throttled(math.ceil(hits[0] + _WINDOW - now))
    if s.VISUAL_JOBS_PER_HOUR > 0 and await crud.jobs_created_last_hour() >= s.VISUAL_JOBS_PER_HOUR:
        raise Throttled(_GLOBAL_RETRY_AFTER)
    hits.append(now)


async def prepare(user_id: str, car: dict, variant_key: str | None) -> tuple[VisualDescriptor, bool]:
    make, model, year = car.get("make"), car.get("model"), _year(car)
    if variant_key is not None and not (catalog.is_allowed(variant_key) and catalog.variant_matches_car(variant_key, make, model)):
        raise VariantMismatch(variant_key)
    cands = catalog.candidates(make, model, year)
    try:
        choice = variant_key if variant_key in cands else await _stored_choice(user_id, car, cands)
        if variant_key is not None and choice == variant_key:
            await crud.set_choice(user_id, str(car.get("id")), variant_key)
        res = catalog.resolve(make, model, year, choice)
        if res.kind != "selected":
            return _fallback("choose_variant" if res.kind == "choose" else "unknown_variant", res.options), False
        row = await crud.get_visual(res.variant_key)
        created = False
        if row is None:
            await _check_throttles(user_id)
            row, created = await crud.ensure_job(res.variant_key)
        return await _from_row(res.variant_key, row), created
    except (DatabaseUnavailable, Throttled):
        raise
    except Exception as exc:
        return await _unavailable(exc), False
