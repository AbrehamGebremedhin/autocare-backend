"""Vehicle visuals worker: claims queued jobs, generates artwork, parks a normalized draft in review.

Run as `python -m app.workers.vehicle_visuals`. Nothing is ever published here; an operator approves later.
Logs carry variant keys, stable codes and exception class names only: never shas, paths or image bytes.
"""
import asyncio
import random
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

if __name__ == "__main__":  # running the file directly instead of with -m
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.CRUD import vehicle_visual_crud as crud  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.db import pool  # noqa: E402
from app.services.visual_generators import Generator, GeneratorError, build_generator  # noqa: E402
from app.services.visual_image import InvalidArtwork, normalize, quality_problem  # noqa: E402
from app.services.visual_store import VisualStore  # noqa: E402
from app.utils.logger import get_logger_instance  # noqa: E402

logger = get_logger_instance("visuals_worker")

LEASE_SECONDS = 300
MAX_ATTEMPTS = 3
INTERNAL_ERROR_DELAY_SECONDS = 60
SWEEP_EVERY_SECONDS = 3600
SWEEP_GRACE_SECONDS = 3600


class _LeaseLost(Exception):
    pass


async def _heartbeat(job_id: str, token: int) -> None:
    """Renew the lease every LEASE_SECONDS/3 and return only when it is lost. A renewal that errors is retried at
    the next beat until the last good one is old enough that the lease may have run out."""
    every = LEASE_SECONDS / 3
    last_good = time.monotonic()
    while True:
        await asyncio.sleep(every)
        try:
            held = await crud.renew_lease(job_id, token, LEASE_SECONDS)
        except Exception as exc:
            await logger.warning(f"visual lease renew failed: {type(exc).__name__}")
            if time.monotonic() - last_good < LEASE_SECONDS - every:
                continue
            held = False
        if not held:
            return
        last_good = time.monotonic()


async def _generate(gen: Generator, job: dict) -> bytes:
    """Run the generator while the heartbeat keeps the lease; a lost lease cancels the generation."""
    work = asyncio.create_task(gen.generate(job["variant_key"], crud.STYLE_VERSION))
    beat = asyncio.create_task(_heartbeat(str(job["id"]), job["fencing_token"]))
    try:
        await asyncio.wait({work, beat}, return_when=asyncio.FIRST_COMPLETED)
        if not work.done():
            raise _LeaseLost
        return work.result()
    finally:
        work.cancel()
        beat.cancel()
        await asyncio.gather(work, beat, return_exceptions=True)


async def _finalize(job: dict, state: str, sha: str | None, error: str | None, seconds: float | None) -> bool:
    return await crud.finalize_job(str(job["id"]), job["fencing_token"], state, sha, error, seconds)


async def _fail_or_retry(job: dict, code: str, seconds: float) -> None:
    if job["attempt_count"] >= MAX_ATTEMPTS:
        await _finalize(job, "failed", None, code, None)
    else:
        await _finalize(job, "queued", None, code, seconds)


async def _process(job: dict, gen: Generator, store: VisualStore, jitter: Callable[[], float]) -> None:
    variant = job["variant_key"]
    if job["attempt_count"] > MAX_ATTEMPTS:  # the SQL claim has no attempt cap
        await _finalize(job, "failed", None, "max_attempts", None)
        return
    if gen.charged:
        if await crud.provider_calls_last_day() >= get_settings().VISUAL_PROVIDER_DAILY_BUDGET:
            await logger.warning(f"visual provider budget exhausted: {variant} budget_exhausted")
            await _finalize(job, "failed", None, "budget_exhausted", None)
            return
        await crud.record_provider_call(variant)  # before the call: a timeout may still have been charged
    try:
        data = await _generate(gen, job)
    except GeneratorError as exc:
        delay = min(2 ** job["attempt_count"] * 30, 900) * (1 + 0.25 * jitter())
        await logger.warning(f"visual generation failed: {variant} {exc.code}")
        await _fail_or_retry(job, exc.code, delay)
        return
    try:
        webp, sha, _, _ = await asyncio.to_thread(normalize, data)  # CPU bound: never on the event loop
    except InvalidArtwork as exc:
        await logger.warning(f"visual artwork rejected: {variant} {exc.code}")
        await _finalize(job, "failed", None, exc.code, None)
        return
    await asyncio.to_thread(store.write_draft, variant, crud.STYLE_VERSION, job["fencing_token"], webp)
    if not await _finalize(job, "review", sha, None, None):
        await logger.warning(f"visual finalize skipped, lease was reclaimed: {variant}")  # draft is swept later
        return
    if gen.charged and get_settings().VISUAL_AUTO_APPROVE:
        await _auto_publish(variant, store, webp, sha)


async def _auto_publish(variant: str, store: VisualStore, webp: bytes, sha: str) -> None:
    """Publish a finished draft like the operator's approve would, if it passes the quality check. Anything else
    stays in review for a human; this never raises, so a failure here cannot lose the draft."""
    try:
        problem = await asyncio.to_thread(quality_problem, webp)
        if problem:
            await logger.warning(f"visual auto-approve declined: {variant} {problem}")
            return
        row = await crud.get_visual(variant)
        if row is None or row["state"] != "review" or row["draft_sha256"] != sha:
            return  # someone (an operator, a requeue) moved it meanwhile
        await asyncio.to_thread(store.put_object, sha, webp)  # object first, activate second
        if await crud.publish(row["id"], row["published_revision"], sha, 1024, 420):
            try:
                store.delete_draft(store.draft_rel(variant, crud.STYLE_VERSION, row["fencing_token"]))
            except (ValueError, OSError):
                pass  # the sweep reclaims it
            await logger.info(f"visual auto-approved: {variant}")
    except Exception as exc:
        await logger.error(f"visual auto-approve failed, left in review: {variant} {type(exc).__name__}")


async def run_once(
    gen: Generator,
    store: VisualStore,
    now: Callable[[], float] = time.time,
    jitter: Callable[[], float] = random.random,
) -> bool:
    """Handle one due job. True if a job was claimed. Never raises except CancelledError; a cancelled run leaves
    its lease to expire, after which the SQL claim hands the job out again."""
    try:
        job = await crud.claim_job(LEASE_SECONDS)
    except Exception as exc:
        await logger.error(f"visual claim failed: {type(exc).__name__}")
        return False
    if job is None:
        return False
    started = now()
    try:
        await _process(job, gen, store, jitter)
    except _LeaseLost:
        await logger.warning(f"visual lease lost, work abandoned: {job['variant_key']}")
    except Exception as exc:
        await logger.error(f"visual job failed unexpectedly: {job['variant_key']} {type(exc).__name__}")
        try:
            await _fail_or_retry(job, "internal_error", INTERNAL_ERROR_DELAY_SECONDS)
        except Exception as inner:
            await logger.error(f"visual job could not be released: {type(inner).__name__}")
    await logger.info(f"visual job handled: {job['variant_key']} in {now() - started:.1f}s")
    return True


def _install_stop(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows event loops
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))


async def _sweep(store: VisualStore) -> None:
    try:
        keep_drafts = set()
        for v, s, t in await crud.referenced_drafts():
            try:
                keep_drafts.add(store.draft_rel(v, s, t))
            except ValueError:  # such a row cannot have a valid draft path; the rest of the sweep still runs
                await logger.warning(f"visual review row has no valid draft path: {v} bad_draft_key")
        removed = await asyncio.to_thread(
            store.sweep, older_than_seconds=SWEEP_GRACE_SECONDS, keep_shas=await crud.referenced_shas(),
            keep_drafts=keep_drafts)
        if removed:
            await logger.info(f"visual store sweep removed {removed} files")
    except Exception as exc:
        await logger.warning(f"visual store sweep failed: {type(exc).__name__}")


async def main() -> None:
    stop = asyncio.Event()
    _install_stop(stop)
    await pool.init_pool()
    try:
        gen, store = build_generator(), VisualStore()
        last_sweep = float("-inf")
        while not stop.is_set():
            handled = await run_once(gen, store)
            if time.monotonic() - last_sweep >= SWEEP_EVERY_SECONDS:
                last_sweep = time.monotonic()
                await _sweep(store)
            if not handled and not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), get_settings().VISUAL_POLL_SECONDS)
                except TimeoutError:
                    pass
    finally:
        await pool.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
