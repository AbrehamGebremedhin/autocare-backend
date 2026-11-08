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
        await asyncio.sleep(0)  # a real query yields; this is where concurrent calls interleave
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


def test_unsignable_url_is_fallback(world, monkeypatch, caplog):
    world.row(ECHO, "ready", 1, SHA)
    monkeypatch.setattr(vv, "sign_url", lambda sha: ("", ""))
    with caplog.at_level(logging.INFO):
        d = run(vv.get_descriptor("u1", ECHO_CAR))
    assert d.reason == "temporarily_unavailable" and d.artwork is None
    assert "could not sign" in caplog.text and "unavailable:" not in caplog.text


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


def test_never_published_failed_is_fallback(world, caplog):
    world.row(ECHO, "failed")
    with caplog.at_level(logging.INFO):
        d = run(vv.get_descriptor("u1", ECHO_CAR))
    assert d.reason == "temporarily_unavailable" and d.variant_key is None
    assert "unavailable:" not in caplog.text and "missing from the store" not in caplog.text


def test_choose_variant_carries_labels_only(world):
    d = run(vv.get_descriptor("u1", CIVIC_CAR))
    assert (d.state, d.reason) == ("fallback", "choose_variant")
    assert {(o.variant_key, o.label) for o in d.options} == {(CIVIC, catalog.label(CIVIC)), (COUPE, "Honda Civic Coupe")}
    assert set(d.to_dict()["options"][0]) == {"variant_key", "label"}
    assert d.variant_key is None


def test_car_not_in_catalog_asks_for_a_body_style(world):
    d = run(vv.get_descriptor("u1", LADA_CAR))
    assert (d.state, d.reason, d.variant_key) == ("fallback", "choose_variant", None)
    assert [o.label for o in d.options][:2] == ["Lada Niva 1990 Sedan", "Lada Niva 1990 Hatchback"]
    assert len(d.options) == 8


def test_unusable_car_is_unknown_variant(world):
    d = run(vv.get_descriptor("u1", {**LADA_CAR, "make": "!!!"}))
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
    assert run(vv.prepare("u1", LADA_CAR, None))[0].reason == "choose_variant"
    assert world.ensure_calls == 0


def test_choosing_a_body_style_queues_one_shared_job(world):
    key = "auto-lada-niva-1990-suv"
    d, created = run(vv.prepare("u1", LADA_CAR, key))
    assert (d.state, d.variant_key, created) == ("pending", key, True)
    assert run(vv.get_descriptor("u1", LADA_CAR)).variant_key == key  # the choice is remembered
    d2, created2 = run(vv.prepare("u2", {**LADA_CAR, "id": "car-9"}, key))
    assert d2.variant_key == key and created2 is False  # same variant, no second job
    with pytest.raises(vv.VariantMismatch):
        run(vv.prepare("u1", LADA_CAR, "auto-lada-niva-1990-boat"))
    with pytest.raises(vv.VariantMismatch):
        run(vv.prepare("u1", LADA_CAR, "auto-lada-2110-1990-suv"))


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


def test_stored_choice_cleared_when_no_longer_candidate(world):
    world.choices[("u1", "car-2")] = FUTURE
    d = run(vv.get_descriptor("u1", CIVIC_CAR))
    assert d.reason == "choose_variant" and ("u1", "car-2") not in world.choices


def test_programming_error_in_pure_logic_propagates(world, monkeypatch):
    world.row(ECHO, "queued")

    def typo(*a, **k):
        raise AttributeError("typo")

    monkeypatch.setattr(catalog, "resolve", typo)
    with pytest.raises(AttributeError):
        run(vv.get_descriptor("u1", ECHO_CAR))
    with pytest.raises(AttributeError):
        run(vv.prepare("u1", ECHO_CAR, None))


def test_programming_error_in_descriptor_construction_propagates(world, monkeypatch):
    world.row(ECHO, "queued")
    monkeypatch.setattr(vv, "_pending", lambda *a, **k: {}["boom"])
    with pytest.raises(KeyError):
        run(vv.get_descriptor("u1", ECHO_CAR))


def test_store_and_sign_failures_degrade(world, monkeypatch, caplog):
    world.row(ECHO, "ready", 1, SHA)

    def bad(sha):
        raise OSError("disk")

    monkeypatch.setattr(vv, "sign_url", bad)
    with caplog.at_level(logging.INFO):
        d = run(vv.get_descriptor("u1", ECHO_CAR))
    assert d.reason == "temporarily_unavailable" and "unavailable: OSError" in caplog.text


def _spy_writes(world, monkeypatch):
    calls = []
    monkeypatch.setattr(vv, "sign_url", lambda sha: calls.append("sign") or ("https://x/y", "t"))
    orig_set = crud.set_choice

    async def set_choice(*a):
        calls.append("set_choice")
        await orig_set(*a)

    monkeypatch.setattr(crud, "set_choice", set_choice)
    return calls


def test_get_for_unowned_car_raises_and_signs_nothing(world, monkeypatch):
    world.row(ECHO, "ready", 1, SHA)
    world.owned = False
    calls = _spy_writes(world, monkeypatch)
    with pytest.raises(vv.CarNotOwned):
        run(vv.get_descriptor("u1", ECHO_CAR))
    assert calls == []


@pytest.mark.parametrize("key", [None, ECHO])
def test_prepare_for_unowned_car_raises_before_any_write(world, monkeypatch, key):
    world.row(ECHO, "ready", 1, SHA)
    world.owned = False
    calls = _spy_writes(world, monkeypatch)
    with pytest.raises(vv.CarNotOwned):
        run(vv.prepare("u1", ECHO_CAR, key))
    with pytest.raises(vv.CarNotOwned):
        run(vv.prepare("u1", CIVIC_CAR, COUPE))
    assert calls == [] and world.ensure_calls == 0 and world.choices == {}


def test_owner_check_runs_once_per_request_even_with_a_variant_key(world, monkeypatch):
    seen = []
    orig = crud.owned_car

    async def counting(u, c):
        seen.append(c)
        return await orig(u, c)

    monkeypatch.setattr(crud, "owned_car", counting)
    run(vv.prepare("u1", CIVIC_CAR, COUPE))
    run(vv.get_descriptor("u1", CIVIC_CAR))
    assert seen == ["car-2", "car-2"]


def test_owner_pool_outage_propagates(world, monkeypatch):
    from app.db.pool import DatabaseUnavailable

    async def down(u, c):
        raise DatabaseUnavailable("down")

    monkeypatch.setattr(crud, "owned_car", down)
    with pytest.raises(DatabaseUnavailable):
        run(vv.get_descriptor("u1", ECHO_CAR))
    with pytest.raises(DatabaseUnavailable):
        run(vv.prepare("u1", ECHO_CAR, ECHO))


def test_concurrent_throttle_checks_admit_exactly_the_limit(world):
    world.settings.VISUAL_PREPARE_PER_HOUR = 1

    async def many():
        return await asyncio.gather(*[vv._check_throttles("u1") for _ in range(10)], return_exceptions=True)

    out = run(many())
    throttled = [r for r in out if isinstance(r, vv.Throttled)]
    assert sum(r is None for r in out) == 1 and len(throttled) == 9
    assert all(t.retry_after >= 1 for t in throttled)


def test_attempt_rolled_back_when_global_cap_throws(world):
    world.settings.VISUAL_PREPARE_PER_HOUR = 1
    world.settings.VISUAL_JOBS_PER_HOUR = 5
    world.jobs_last_hour = 5
    with pytest.raises(vv.Throttled) as exc:
        run(vv._check_throttles("u1"))
    assert exc.value.retry_after == 300 and vv._prepare_hits.get("u1", []) == []
    world.jobs_last_hour = 0
    run(vv._check_throttles("u1"))  # the user's one slot was not consumed


def test_attempt_rolled_back_when_global_count_fails(world, monkeypatch):
    world.settings.VISUAL_PREPARE_PER_HOUR = 1

    async def broken():
        raise OSError("db")

    monkeypatch.setattr(crud, "jobs_created_last_hour", broken)
    with pytest.raises(vv._Unavailable):
        run(vv._check_throttles("u1"))
    assert vv._prepare_hits.get("u1", []) == []


def test_stale_entries_of_other_users_are_pruned(world):
    for n in range(5):
        run(vv._check_throttles(f"old{n}"))
    assert len(vv._prepare_hits) == 5
    world.now += 3601
    run(vv._check_throttles("fresh"))
    assert list(vv._prepare_hits) == ["fresh"]


def test_zero_user_limit_records_nothing(world):
    world.settings.VISUAL_PREPARE_PER_HOUR = 0
    run(vv._check_throttles("u1"))
    assert "u1" not in vv._prepare_hits
