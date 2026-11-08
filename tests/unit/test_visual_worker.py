"""The visuals worker's run_once and main, over fakes: no database, no network, no provider."""
import asyncio
import io
import threading

import pytest
from PIL import Image, ImageDraw

from app.services import visual_image
from app.services.visual_generators import GeneratorError
from app.services.visual_store import VisualStore
from app.workers import vehicle_visuals as w

pytestmark = pytest.mark.usefixtures("no_network")


def _png() -> bytes:
    img = Image.new("RGBA", (600, 200), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle((20, 20, 580, 180), fill=(200, 30, 30, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _bomb() -> bytes:
    buf = io.BytesIO()
    Image.new("1", (6000, 6000)).save(buf, "PNG")  # a few KB on the wire, 36M pixels decoded
    return buf.getvalue()


class FakeGen:
    def __init__(self, result=None, charged=False, delay=0.0):
        self.result = result
        self.charged = charged
        self.delay = delay
        self.calls = []

    async def generate(self, variant_key, style_version):
        self.calls.append((variant_key, style_version))
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeCrud:
    def __init__(self, job=None):
        self.job = job
        self.finalized = []
        self.finalize_result = True
        self.renew_results = None  # None: always True
        self.renewals = 0
        self.provider_calls = 0
        self.recorded = []
        self.order = []

    async def claim_job(self, lease_seconds):
        job, self.job = self.job, None
        return job

    async def renew_lease(self, job_id, token, lease_seconds):
        self.renewals += 1
        return True if self.renew_results is None else self.renew_results(self.renewals)

    async def finalize_job(self, job_id, token, state, sha, error, seconds):
        self.finalized.append((state, sha, error, seconds))
        return self.finalize_result

    async def provider_calls_last_day(self):
        return self.provider_calls

    async def record_provider_call(self, variant_key):
        self.recorded.append(variant_key)
        self.order.append("record")

    async def referenced_shas(self):
        return set()

    review_drafts = ()

    async def referenced_drafts(self):
        return set(self.review_drafts)


def _job(attempt=1, token=7, variant="toyota-echo"):
    return {"id": "11111111-1111-4111-8111-111111111111", "variant_key": variant, "fencing_token": token,
            "attempt_count": attempt}


@pytest.fixture
def crud(monkeypatch):
    fake = FakeCrud(_job())
    for name in ("claim_job", "renew_lease", "finalize_job", "provider_calls_last_day", "record_provider_call",
                 "referenced_shas", "referenced_drafts"):
        monkeypatch.setattr(w.crud, name, getattr(fake, name))
    return fake


@pytest.fixture
def store(tmp_path):
    return VisualStore(tmp_path / "visuals")


def _drafts(store):
    return list((store._base() / "drafts").rglob("*.webp")) if (store._base() / "drafts").exists() else []


async def test_nothing_due_returns_false(crud, store):
    crud.job = None
    assert await w.run_once(FakeGen(b""), store) is False
    assert crud.finalized == []


async def test_happy_path_parks_a_draft_in_review(crud, store):
    assert await w.run_once(FakeGen(_png()), store) is True
    (state, sha, error, seconds), = crud.finalized
    assert (state, error, seconds) == ("review", None, None)
    (draft,) = _drafts(store)
    assert draft.relative_to(store._base()).as_posix() == "drafts/toyota-echo/workshop-v1/7.webp"
    webp = draft.read_bytes()
    assert visual_image.validate_published(webp)[0] == sha
    assert crud.recorded == []  # an uncharged generator is never recorded


async def test_generator_error_retries_with_growing_delay_then_fails(crud, store):
    gen = FakeGen(GeneratorError("no_source"))
    delays = []
    for attempt in (1, 2, 3):
        crud.job = _job(attempt=attempt)
        assert await w.run_once(gen, store, jitter=lambda: 0.5) is True
        delays.append(crud.finalized[-1])
    assert delays[0] == ("queued", None, "no_source", 60 * 1.125)
    assert delays[1] == ("queued", None, "no_source", 120 * 1.125)
    assert delays[2] == ("failed", None, "no_source", None)
    assert _drafts(store) == []


async def test_backoff_is_capped_at_900_seconds(crud, store, monkeypatch):
    monkeypatch.setattr(w, "MAX_ATTEMPTS", 10)
    crud.job = _job(attempt=9)
    await w.run_once(FakeGen(GeneratorError("x")), store, jitter=lambda: 1.0)
    assert crud.finalized[-1][3] == 900 * 1.25


async def test_attempt_beyond_the_cap_fails_without_calling_the_generator(crud, store):
    crud.job = _job(attempt=4)
    gen = FakeGen(_png())
    assert await w.run_once(gen, store) is True
    assert crud.finalized == [("failed", None, "max_attempts", None)]
    assert gen.calls == []


async def test_budget_exhausted_never_calls_the_generator(crud, store, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "VISUAL_PROVIDER_DAILY_BUDGET", 5)
    crud.provider_calls = 5
    gen = FakeGen(_png(), charged=True)
    assert await w.run_once(gen, store) is True
    assert gen.calls == [] and crud.recorded == []
    assert crud.finalized == [("failed", None, "budget_exhausted", None)]


async def test_budget_exhausted_is_logged_with_variant_and_code_only(crud, store, monkeypatch):
    logged = []

    async def warn(message):
        logged.append(message)

    monkeypatch.setattr(w.logger, "warning", warn)
    crud.provider_calls = 10**6
    await w.run_once(FakeGen(_png(), charged=True), store)
    (line,) = logged
    assert "budget_exhausted" in line and "toyota-echo" in line
    assert str(store._base()) not in line and "sha" not in line.lower()


async def test_worker_sweep_keeps_review_drafts_and_removes_others(crud, store):
    import os

    kept = store.write_draft("toyota-echo", "workshop-v1", 7, b"d")
    gone = store.write_draft("other-car", "workshop-v1", 2, b"d")
    lookalike = store.write_draft("toyota-echo", "workshop-v1", 8, b"d")
    for rel in (kept, gone, lookalike):
        os.utime(store._base() / rel, (1, 1))  # far older than the grace period
    crud.review_drafts = {("toyota-echo", "workshop-v1", 7)}
    await w._sweep(store)
    assert (store._base() / kept).exists()
    assert not (store._base() / gone).exists() and not (store._base() / lookalike).exists()


async def test_one_bad_review_row_does_not_abort_the_sweep(crud, store, monkeypatch):
    import os

    kept = store.write_draft("toyota-echo", "workshop-v1", 7, b"d")
    gone = store.write_draft("other-car", "workshop-v1", 2, b"d")
    sha = "a" * 64
    obj_dir = store._base() / "objects"
    obj_dir.mkdir(parents=True)
    obj = obj_dir / f"{sha}.webp"
    obj.write_bytes(b"x")
    for path in (store._base() / kept, store._base() / gone, obj):
        os.utime(path, (1, 1))
    crud.review_drafts = {("bad/../key", "workshop-v1", 1), ("toyota-echo", "workshop-v1", 7)}
    logged = []

    async def warn(message):
        logged.append(message)

    monkeypatch.setattr(w.logger, "warning", warn)
    await w._sweep(store)
    assert (store._base() / kept).exists()
    assert not (store._base() / gone).exists() and not obj.exists()
    bad = [m for m in logged if "bad_draft_key" in m]
    assert len(bad) == 1 and "bad/../key" in bad[0] and str(store._base()) not in bad[0]
    assert not any("sweep failed" in m for m in logged)


async def test_charged_call_is_recorded_before_generate_and_survives_a_timeout(crud, store, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "VISUAL_PROVIDER_DAILY_BUDGET", 5)
    crud.provider_calls = 4
    gen = FakeGen(GeneratorError("provider_timeout"), charged=True)
    original = gen.generate

    async def ordered(variant_key, style_version):
        crud.order.append("generate")
        return await original(variant_key, style_version)

    gen.generate = ordered
    await w.run_once(gen, store, jitter=lambda: 0.0)
    assert crud.order == ["record", "generate"]
    assert crud.recorded == ["toyota-echo"]
    assert crud.finalized == [("queued", None, "provider_timeout", 60.0)]


@pytest.mark.parametrize("data, code", [(b"not an image at all", "not_image"), (None, "too_large")])
async def test_invalid_image_fails_with_its_code_and_writes_no_draft(crud, store, data, code):
    assert await w.run_once(FakeGen(_bomb() if data is None else data), store) is True
    assert crud.finalized == [("failed", None, code, None)]
    assert _drafts(store) == []


async def test_normalize_runs_off_the_event_loop(crud, store, monkeypatch):
    seen = []
    real = w.normalize

    def spy(data):
        seen.append(threading.current_thread())
        return real(data)

    monkeypatch.setattr(w, "normalize", spy)
    await w.run_once(FakeGen(_png()), store)
    assert seen and seen[0] is not threading.main_thread()


async def test_heartbeat_renews_during_a_slow_generator(crud, store, monkeypatch):
    monkeypatch.setattr(w, "LEASE_SECONDS", 0.3)  # renews every 0.1 s
    await w.run_once(FakeGen(_png(), delay=0.45), store)
    assert crud.renewals >= 2
    assert crud.finalized[-1][0] == "review"


async def test_lost_lease_stops_the_work_and_finalizes_nothing(crud, store, monkeypatch):
    monkeypatch.setattr(w, "LEASE_SECONDS", 0.3)
    crud.renew_results = lambda n: False
    gen = FakeGen(_png(), delay=5)
    assert await asyncio.wait_for(w.run_once(gen, store), 2) is True
    assert crud.finalized == []
    assert _drafts(store) == []


async def test_stale_token_finalize_writes_nothing_further(crud, store):
    crud.finalize_result = False
    assert await w.run_once(FakeGen(_png()), store) is True
    assert len(crud.finalized) == 1  # one attempt, no retry-style second write


async def test_unexpected_exception_requeues_as_internal_error(crud, store):
    assert await w.run_once(FakeGen(RuntimeError("secret detail")), store) is True
    assert crud.finalized == [("queued", None, "internal_error", 60)]


async def test_unexpected_exception_at_the_cap_fails(crud, store):
    crud.job = _job(attempt=3)
    await w.run_once(FakeGen(RuntimeError("x")), store)
    assert crud.finalized == [("failed", None, "internal_error", None)]


async def test_unexpected_exception_log_has_class_name_only(crud, store, monkeypatch):
    logged = []

    async def warn(message):
        logged.append(message)

    monkeypatch.setattr(w.logger, "warning", warn)
    monkeypatch.setattr(w.logger, "error", warn)
    await w.run_once(FakeGen(RuntimeError("secret detail")), store)
    assert logged and all("secret detail" not in m for m in logged)
    assert any("RuntimeError" in m for m in logged)


async def test_claim_failure_does_not_raise(crud, store, monkeypatch):
    async def down(lease_seconds):
        raise ConnectionError("db down")

    monkeypatch.setattr(w.crud, "claim_job", down)
    assert await w.run_once(FakeGen(b""), store) is False


async def test_cancellation_propagates_and_leaves_the_lease_alone(crud, store):
    task = asyncio.create_task(w.run_once(FakeGen(_png(), delay=5), store))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert crud.finalized == []


async def test_main_stops_on_signal_and_closes_the_pool(monkeypatch, store):
    import signal

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    events = []

    async def init():
        events.append("init")

    async def close():
        events.append("close")

    async def fake_run_once(gen, st):
        events.append("run")
        signal.raise_signal(signal.SIGINT)
        return False  # idle: main would sleep the poll interval, but the signal cuts it short

    monkeypatch.setattr(w.pool, "init_pool", init)
    monkeypatch.setattr(w.pool, "close_pool", close)
    monkeypatch.setattr(w, "run_once", fake_run_once)
    monkeypatch.setattr(w, "build_generator", lambda: FakeGen(b""))
    monkeypatch.setattr(w, "VisualStore", lambda: store)
    monkeypatch.setattr(w.crud, "referenced_shas", FakeCrud().referenced_shas)
    monkeypatch.setattr(w.crud, "referenced_drafts", FakeCrud().referenced_drafts)
    try:
        await asyncio.wait_for(w.main(), 5)
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    assert events == ["init", "run", "close"]
