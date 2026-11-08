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
from app.services import body_styles, vehicle_catalog as catalog
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


class CarNotOwned(Exception):
    """The car is not (or no longer) the user's; the route answers it like an inaccessible car."""


class _Unavailable(Exception):
    """An I/O call on the visuals tables, the store or signing failed; the caller degrades to a fallback."""


class Throttled(Exception):
    def __init__(self, retry_after: int):
        super().__init__(f"throttled, retry after {retry_after}s")
        self.retry_after = max(1, int(retry_after))


def _now() -> float:
    return time()


def _fallback(reason: str, options: list[tuple[str, str]] | None = None) -> VisualDescriptor:
    opts = [VisualOption(variant_key=k, label=lbl) for k, lbl in options] if reason == "choose_variant" and options else None
    return VisualDescriptor(state="fallback", preset=None, reason=reason, options=opts)


def _pending(variant_key: str | None, ms: int = _PENDING_MS) -> VisualDescriptor:
    return VisualDescriptor(state="pending", variant_key=variant_key, style_version=crud.STYLE_VERSION, retry_after_ms=ms)


async def _degraded(exc: Exception) -> _Unavailable:
    await logger.warning(f"vehicle visuals unavailable: {type(exc).__name__}")
    return _Unavailable()


async def _io(awaitable):
    """Await one visuals-table call. Pool outages and cancellation propagate; any other failure becomes _Unavailable."""
    try:
        return await awaitable
    except DatabaseUnavailable:
        raise
    except Exception as exc:
        raise await _degraded(exc) from None


async def _from_row(variant_key: str, row: dict | None) -> VisualDescriptor:
    if row is None:
        return _pending(variant_key)
    sha = row["sha256"]
    if row["published_revision"] >= 1 and sha:
        try:
            present = VisualStore().has_object(sha)
            url, expires = sign_url(sha) if present else ("", "")
        except Exception as exc:
            raise await _degraded(exc) from None
        if url:
            return VisualDescriptor(
                state="ready", variant_key=variant_key, style_version=crud.STYLE_VERSION,
                artwork=Artwork(sha256=sha, download_url=url, expires_at=expires))
        if present:
            await logger.warning(f"visual {variant_key}: could not sign a download url")
        else:
            await logger.warning(f"visual {variant_key}: published artwork is missing from the store")
        return _fallback("temporarily_unavailable")
    if row["state"] == "review":
        return _pending(variant_key, _REVIEW_MS)
    if row["state"] in ("queued", "generating"):
        return _pending(variant_key)
    return _fallback("temporarily_unavailable")


def _year(car: dict) -> int | None:
    try:
        return int(car.get("year"))
    except (TypeError, ValueError):
        return None


async def _require_owner(user_id: str, car: dict) -> str:
    """One ownership check per request, before anything that could sign a url or write. Pool outages propagate."""
    car_id = str(car["id"])
    if await crud.owned_car(user_id, car_id) is None:
        raise CarNotOwned(car_id)
    return car_id


async def _stored_choice(user_id: str, car_id: str, cands: list[str]) -> str | None:
    stored = await _io(crud.get_choice(user_id, car_id))
    if stored is not None and stored not in cands:
        await _io(crud.clear_choice(user_id, car_id))
        return None
    return stored


def _non_selected(res: catalog.Resolution) -> VisualDescriptor:
    return _fallback("choose_variant" if res.kind == "choose" else "unknown_variant", res.options)


async def _styles(make: str, model: str, year: int | None):
    """Body styles to build variants from: () when the catalog already covers the car (or it is unusable), the
    cached styles, or None when this car has not been looked up yet."""
    if catalog.candidates(make, model, year) or not catalog.usable(make, model, year):
        return ()
    return await _io(body_styles.cached(make, model, year))


async def get_descriptor(user_id: str, car: dict) -> VisualDescriptor:
    car_id = await _require_owner(user_id, car)
    make, model, year = car.get("make"), car.get("model"), _year(car)
    try:
        styles = await _styles(make, model, year)
        if styles is None:
            return _pending(None)  # not looked up yet; prepare does that
        cands = catalog.candidates(make, model, year, styles)
        res = catalog.resolve(make, model, year, await _stored_choice(user_id, car_id, cands), styles)
        if res.kind != "selected":
            return _non_selected(res)
        return await _from_row(res.variant_key, await _io(crud.get_visual(res.variant_key)))
    except _Unavailable:
        return _fallback("temporarily_unavailable")


async def _check_throttles(user_id: str) -> None:
    """Record the attempt synchronously (no await between the read and the append), then check the global cap."""
    now = _now()
    for uid in [u for u, hits in _prepare_hits.items() if all(h <= now - _WINDOW for h in hits)]:
        del _prepare_hits[uid]
    s = get_settings()
    hits = None
    if s.VISUAL_PREPARE_PER_HOUR > 0:
        hits = _prepare_hits.setdefault(user_id, [])
        hits[:] = [h for h in hits if h > now - _WINDOW]
        if len(hits) >= s.VISUAL_PREPARE_PER_HOUR:
            raise Throttled(math.ceil(min(hits) + _WINDOW - now))
        hits.append(now)
    try:
        if s.VISUAL_JOBS_PER_HOUR > 0 and await _io(crud.jobs_created_last_hour()) >= s.VISUAL_JOBS_PER_HOUR:
            raise Throttled(_GLOBAL_RETRY_AFTER)
    except BaseException:
        if hits is not None and now in hits:
            hits.remove(now)  # this attempt created nothing
        raise


async def prepare(user_id: str, car: dict, variant_key: str | None) -> tuple[VisualDescriptor, bool]:
    car_id = await _require_owner(user_id, car)
    make, model, year = car.get("make"), car.get("model"), _year(car)
    if variant_key is not None and not (catalog.is_allowed(variant_key) and catalog.variant_matches_car(variant_key, make, model)):
        raise VariantMismatch(variant_key)
    try:
        throttled = False  # one throttle check per call, whether it asks the LLM, creates a job, or both
        styles = await _styles(make, model, year)
        if styles is None:
            await _check_throttles(user_id)
            throttled = True
            styles = await body_styles.lookup(make, model, year)
            if styles is None:
                raise _Unavailable()  # the LLM failed: nothing stored, try again later
        cands = catalog.candidates(make, model, year, styles)
        if variant_key in cands:
            choice = variant_key
            await _io(crud.set_choice(user_id, car_id, variant_key))
        else:
            choice = await _stored_choice(user_id, car_id, cands)
        res = catalog.resolve(make, model, year, choice, styles)
        if res.kind != "selected":
            return _non_selected(res), False
        row = await _io(crud.get_visual(res.variant_key))
        created = False
        if row is None:
            if not throttled:
                await _check_throttles(user_id)
            row, created = await _io(crud.ensure_job(res.variant_key))
        return await _from_row(res.variant_key, row), created
    except _Unavailable:
        return _fallback("temporarily_unavailable"), False
