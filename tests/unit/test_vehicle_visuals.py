import asyncio
import logging
from types import SimpleNamespace

import asyncpg
import pytest

from app.CRUD import vehicle_visual_crud as crud
from app.services import vehicle_catalog as catalog
from app.services import vehicle_visuals as vv

SHA = "a" * 64
ECHO = "toyota-echo-2001-sedan"
CIVIC = "honda-civic-2008-sedan"
COUPE = "honda-civic-2008-coupe"
FUTURE = "honda-civic-2015-hatch"
ECHO_CAR = {"id": "car-1", "make": "Toyota", "model": "Echo", "year": 2001}
CIVIC_CAR = {"id": "car-2", "make": "Honda", "model": "Civic", "year": 2008}
LADA_CAR = {"id": "car-3", "make": "Lada", "model": "Niva", "year": 1990}


class World:
    """In-test stand-in for the visuals tables, the store and the settings."""

    def __init__(self):
        self.rows = {}
        self.choices = {}
        self.owned = True
        self.objects = {SHA}
        self.ensure_calls = 0
        self.jobs_last_hour = 0
        self.now = 1_000_000.0
        self.settings = SimpleNamespace(VISUAL_PREPARE_PER_HOUR=20, VISUAL_JOBS_PER_HOUR=60)
        self.boom = None

    def row(self, variant_key, state="queued", rev=0, sha=None):
        self.rows[variant_key] = {"variant_key": variant_key, "state": state, "published_revision": rev, "sha256": sha}


@pytest.fixture
def world(monkeypatch):
    w = World()
    base = catalog._variants()
    extra = dict(base)
    extra[COUPE] = {**base[CIVIC], "key": COUPE, "label": "Honda Civic Coupe", "ambiguous": False}
    extra[FUTURE] = {**base[CIVIC], "key": FUTURE, "label": "Honda Civic Hatch", "years": [2012, 2016], "ambiguous": False}
    monkeypatch.setattr(catalog, "_variants", lambda: extra)

    def check():
        if w.boom:
            raise w.boom

    async def get_visual(k):
        check()
        return w.rows.get(k)

    async def ensure_job(k, provenance="curated"):
        check()
        w.ensure_calls += 1
        if k in w.rows:
            return w.rows[k], False
        w.row(k)
        return w.rows[k], True

    async def jobs_created_last_hour():
        return w.jobs_last_hour

    async def get_choice(u, c):
        return w.choices.get((u, c))

    async def set_choice(u, c, k):
        w.choices[(u, c)] = k

    async def clear_choice(u, c):
        w.choices.pop((u, c), None)

    async def owned_car(u, c):
        return {"id": c} if w.owned else None

    for fn in (get_visual, ensure_job, jobs_created_last_hour, get_choice, set_choice, clear_choice, owned_car):
        monkeypatch.setattr(crud, fn.__name__, fn)

    class Store:
        def has_object(self, sha):
            return sha in w.objects

    monkeypatch.setattr(vv, "VisualStore", Store)
    monkeypatch.setattr(vv, "sign_url", lambda sha: (f"https://x.test/{sha}.webp?sig=s", "2030-01-01T00:00:00+00:00"))
    monkeypatch.setattr(vv, "get_settings", lambda: w.settings)
    monkeypatch.setattr(vv, "_now", lambda: w.now)
    vv._prepare_hits.clear()
    yield w
    vv._prepare_hits.clear()


def run(coro):
    return asyncio.run(coro)


def test_ready_when_object_exists(world):
    world.row(ECHO, "ready", 1, SHA)
    d = run(vv.get_descriptor("u1", ECHO_CAR))
    assert d.state == "ready" and d.variant_key == ECHO and d.style_version == "workshop-v1"
    assert d.artwork.sha256 == SHA and d.artwork.download_url.startswith("https://")
    assert d.retry_after_ms is None and d.reason is None and d.options is None


@pytest.mark.parametrize("state", ["review", "failed", "queued", "generating"])
def test_published_row_serves_ready_in_any_state(world, state):
    world.row(ECHO, state, 2, SHA)
    assert run(vv.get_descriptor("u1", ECHO_CAR)).state == "ready"


def test_missing_object_is_fallback_and_logs_without_sha_or_url(world, caplog):
    world.row(ECHO, "ready", 1, SHA)
    world.objects.clear()
    with caplog.at_level(logging.INFO):
        d = run(vv.get_descriptor("u1", ECHO_CAR))
    assert (d.state, d.reason, d.artwork, d.variant_key) == ("fallback", "temporarily_unavailable", None, None)
    assert ECHO in caplog.text
    assert SHA not in caplog.text and "https://" not in caplog.text


def test_unsignable_url_is_fallback(world, monkeypatch):
    world.row(ECHO, "ready", 1, SHA)
    monkeypatch.setattr(vv, "sign_url", lambda sha: ("", ""))
    assert run(vv.get_descriptor("u1", ECHO_CAR)).reason == "temporarily_unavailable"


@pytest.mark.parametrize("state,ms", [("queued", 5000), ("generating", 5000), ("review", 30000)])
def test_pending_retry_values(world, state, ms):
    world.row(ECHO, state)
    d = run(vv.get_descriptor("u1", ECHO_CAR))
    assert d.state == "pending" and d.retry_after_ms == ms and 2000 <= ms <= 60000
    assert d.variant_key == ECHO and d.artwork is None


def test_no_row_is_pending_and_get_creates_no_job(world):
    d = run(vv.get_descriptor("u1", ECHO_CAR))
    assert d.state == "pending" and d.retry_after_ms == 5000
    assert world.ensure_calls == 0 and ECHO not in world.rows


def test_never_published_failed_is_fallback(world):
    world.row(ECHO, "failed")
    assert run(vv.get_descriptor("u1", ECHO_CAR)).reason == "temporarily_unavailable"


def test_choose_variant_carries_labels_only(world):
    d = run(vv.get_descriptor("u1", CIVIC_CAR))
    assert (d.state, d.reason) == ("fallback", "choose_variant")
    assert {(o.variant_key, o.label) for o in d.options} == {(CIVIC, catalog.label(CIVIC)), (COUPE, "Honda Civic Coupe")}
    assert set(d.to_dict()["options"][0]) == {"variant_key", "label"}
    assert d.variant_key is None


def test_unknown_variant_has_no_options_or_variant(world):
    d = run(vv.get_descriptor("u1", LADA_CAR))
    assert (d.state, d.reason, d.options, d.variant_key) == ("fallback", "unknown_variant", None, None)


def test_year_as_string_or_garbage(world):
    assert run(vv.get_descriptor("u1", {**ECHO_CAR, "year": "2001"})).state == "pending"
    assert run(vv.get_descriptor("u1", {**ECHO_CAR, "year": "abc"})).reason == "unknown_variant"
    assert run(vv.get_descriptor("u1", {**ECHO_CAR, "year": None})).reason == "unknown_variant"


def test_options_only_for_choose_variant(world):
    world.row(ECHO, "ready", 1, SHA)
    for car in (ECHO_CAR, CIVIC_CAR, LADA_CAR):
        d = run(vv.get_descriptor("u1", car))
        assert (d.options is not None) == (d.reason == "choose_variant")
        assert ("options" in d.to_dict()) == (d.reason == "choose_variant")


def test_undefined_table_degrades_without_raising(world, caplog):
    world.boom = asyncpg.UndefinedTableError("relation vehicle_visuals does not exist")
    with caplog.at_level(logging.INFO):
        assert run(vv.get_descriptor("user-xyz", ECHO_CAR)).reason == "temporarily_unavailable"
        d, created = run(vv.prepare("user-xyz", ECHO_CAR, None))
    assert d.reason == "temporarily_unavailable" and created is False
    assert "UndefinedTableError" in caplog.text and "user-xyz" not in caplog.text


def test_database_unavailable_bubbles(world):
    from app.db.pool import DatabaseUnavailable

    world.boom = DatabaseUnavailable("down")
    with pytest.raises(DatabaseUnavailable):
        run(vv.get_descriptor("u1", ECHO_CAR))


def test_wrong_model_variant_raises(world):
    with pytest.raises(vv.VariantMismatch):
        run(vv.prepare("u1", ECHO_CAR, CIVIC))
    with pytest.raises(vv.VariantMismatch):
        run(vv.prepare("u1", ECHO_CAR, "not-in-catalog"))


def test_year_mismatched_variant_is_ignored(world):
    d, created = run(vv.prepare("u1", CIVIC_CAR, FUTURE))
    assert d.reason == "choose_variant" and created is False
    assert world.choices == {}


def test_prepare_creates_one_job_then_reuses(world):
    d1, c1 = run(vv.prepare("u1", ECHO_CAR, None))
    d2, c2 = run(vv.prepare("u1", ECHO_CAR, None))
    assert (d1.state, c1) == ("pending", True)
    assert (d2.state, c2) == ("pending", False)
    assert world.ensure_calls == 1


def test_prepare_with_choice_persists_and_queues(world):
    d, created = run(vv.prepare("u1", CIVIC_CAR, COUPE))
    assert created and d.variant_key == COUPE and world.choices[("u1", "car-2")] == COUPE
    assert run(vv.get_descriptor("u1", CIVIC_CAR)).variant_key == COUPE


def test_prepare_choose_or_unknown_creates_no_job(world):
    assert run(vv.prepare("u1", CIVIC_CAR, None))[1] is False
    assert run(vv.prepare("u1", LADA_CAR, None))[0].reason == "unknown_variant"
    assert world.ensure_calls == 0


def test_two_users_same_variant_same_sha(world):
    world.row(ECHO, "ready", 1, SHA)
    a = run(vv.get_descriptor("u1", ECHO_CAR))
    b = run(vv.get_descriptor("u2", ECHO_CAR))
    assert a.artwork.sha256 == b.artwork.sha256 == SHA


def test_user_throttle(world):
    world.settings.VISUAL_PREPARE_PER_HOUR = 2
    run(vv.prepare("u1", ECHO_CAR, None))
    world.now += 100
    world.rows.clear()
    run(vv.prepare("u1", ECHO_CAR, None))
    world.now += 100
    world.rows.clear()
    with pytest.raises(vv.Throttled) as exc:
        run(vv.prepare("u1", ECHO_CAR, None))
    assert exc.value.retry_after == 3600 - 200
    run(vv.prepare("u2", ECHO_CAR, None))  # other users unaffected
    world.now += 3600
    world.rows.clear()
    run(vv.prepare("u1", ECHO_CAR, None))  # window slid


def test_reuse_of_existing_row_never_counts(world):
    world.settings.VISUAL_PREPARE_PER_HOUR = 1
    world.row(ECHO, "queued")
    for _ in range(5):
        assert run(vv.prepare("u1", ECHO_CAR, None))[1] is False


def test_user_throttle_off_with_zero(world):
    world.settings.VISUAL_PREPARE_PER_HOUR = 0
    for _ in range(30):
        world.rows.clear()
        run(vv.prepare("u1", ECHO_CAR, None))


def test_global_cap(world):
    world.settings.VISUAL_JOBS_PER_HOUR = 5
    world.jobs_last_hour = 5
    with pytest.raises(vv.Throttled) as exc:
        run(vv.prepare("u1", ECHO_CAR, None))
    assert exc.value.retry_after == 300 and world.ensure_calls == 0
    world.settings.VISUAL_JOBS_PER_HOUR = 0
    assert run(vv.prepare("u1", ECHO_CAR, None))[1] is True


def test_stored_choice_cleared_when_not_owned(world):
    world.choices[("u1", "car-2")] = COUPE
    world.owned = False
    d = run(vv.get_descriptor("u1", CIVIC_CAR))
    assert d.reason == "choose_variant" and ("u1", "car-2") not in world.choices


def test_stored_choice_cleared_when_no_longer_candidate(world):
    world.choices[("u1", "car-2")] = FUTURE
    d = run(vv.get_descriptor("u1", CIVIC_CAR))
    assert d.reason == "choose_variant" and ("u1", "car-2") not in world.choices
