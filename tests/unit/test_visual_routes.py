import hashlib
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import asyncpg
import pytest

from app.CRUD import vehicle_visual_crud as crud
from app.core.config import get_settings
from app.services import vehicle_visuals as vv
from app.services.visual_signing import sign_url
from app.services.visual_store import VisualStore
from app.utils.limiter import limiter

ECHO = "toyota-echo-2001-sedan"
CIVIC = "honda-civic-2008-sedan"
ECHO_CAR = {"id": "car-1", "make": "Toyota", "model": "Echo", "year": 2001}
GENERIC_404 = "File not found"
VISUAL = f"/api/v1/cars/{ECHO_CAR['id']}/visual"
PREPARE = VISUAL + "/prepare"


@pytest.fixture(autouse=True)
def _clean():
    limiter.reset()
    vv._prepare_hits.clear()
    yield
    vv._prepare_hits.clear()


@pytest.fixture
def world(monkeypatch, user_id):
    """USER_ID owns ECHO_CAR; the visuals tables are an in-memory dict."""
    w = SimpleNamespace(rows={})

    async def owned_car(uid, car_id):
        return ECHO_CAR if (uid == user_id and car_id == ECHO_CAR["id"]) else None

    async def get_visual(k):
        return w.rows.get(k)

    async def ensure_job(k, provenance="curated"):
        if k in w.rows:
            return w.rows[k], False
        w.rows[k] = {"variant_key": k, "state": "queued", "published_revision": 0, "sha256": None}
        return w.rows[k], True

    async def none(*a):
        return None

    async def zero():
        return 0

    monkeypatch.setattr(crud, "owned_car", owned_car)
    monkeypatch.setattr(crud, "get_visual", get_visual)
    monkeypatch.setattr(crud, "ensure_job", ensure_job)
    monkeypatch.setattr(crud, "jobs_created_last_hour", zero)
    monkeypatch.setattr(crud, "get_choice", none)
    monkeypatch.setattr(crud, "set_choice", none)
    monkeypatch.setattr(crud, "clear_choice", none)
    return w


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("VISUALS_DIR", str(tmp_path / "visuals"))
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://api.test")
    get_settings.cache_clear()
    yield VisualStore()
    get_settings.cache_clear()


def _put(store, data=b"RIFF....WEBPfake"):
    sha = hashlib.sha256(data).hexdigest()
    store.put_object(sha, data)
    return sha


def _signed_path(sha, **kw):
    url, _ = sign_url(sha, **kw)
    assert url
    parts = urlsplit(url)
    return f"{parts.path}?{parts.query}"


# --- car routes -----------------------------------------------------------------

@pytest.mark.parametrize("method,url", [("get", VISUAL), ("post", PREPARE)])
def test_no_token_is_401(client, world, method, url):
    assert getattr(client, method)(url).status_code == 401


@pytest.mark.parametrize("method,url", [("get", VISUAL), ("post", PREPARE)])
def test_other_users_car_is_404_not_403(client, world, auth_headers_other_user, method, url):
    r = getattr(client, method)(url, headers=auth_headers_other_user)
    assert r.status_code == 404
    assert r.json()["message"] == "Car not found"


def test_get_own_car_is_200_descriptor(client, world, auth_headers):
    r = client.get(VISUAL, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["state"] == "pending" and r.json()["variant_key"] == ECHO
    assert ECHO not in world.rows  # GET never creates a job


@pytest.mark.parametrize("method,url", [("get", VISUAL), ("post", PREPARE)])
def test_service_car_not_owned_is_404(client, world, auth_headers, monkeypatch, method, url):
    async def boom(*a, **k):
        raise vv.CarNotOwned("car-1")

    monkeypatch.setattr(vv, "get_descriptor", boom)
    monkeypatch.setattr(vv, "prepare", boom)
    r = getattr(client, method)(url, headers=auth_headers)
    assert r.status_code == 404 and r.json()["message"] == "Car not found"


def test_prepare_new_job_202_then_reuse_200(client, world, auth_headers):
    first = client.post(PREPARE, json={}, headers=auth_headers)
    second = client.post(PREPARE, json={}, headers=auth_headers)
    assert first.status_code == 202 and second.status_code == 200
    assert first.json() == second.json() and first.json()["state"] == "pending"


def test_prepare_accepts_missing_body_and_confirmed_variant(client, world, auth_headers):
    assert client.post(PREPARE, headers=auth_headers).status_code in (200, 202)
    r = client.post(PREPARE, json={"variant_key": ECHO}, headers=auth_headers)
    assert r.status_code == 200 and r.json()["variant_key"] == ECHO


def test_prepare_variant_of_another_model_is_422_without_the_value(client, world, auth_headers):
    r = client.post(PREPARE, json={"variant_key": CIVIC}, headers=auth_headers)
    assert r.status_code == 422
    assert r.json()["fields"] == ["variant_key"]
    assert CIVIC not in r.text


@pytest.mark.parametrize("body", [{"x": 1}, {"variant_key": 5}, {"variant_key": ""}, {"variant_key": "Bad Key"}])
def test_prepare_bad_bodies_are_422(client, world, auth_headers, body):
    assert client.post(PREPARE, json=body, headers=auth_headers).status_code == 422


def test_prepare_invalid_json_is_422(client, world, auth_headers):
    r = client.post(PREPARE, content=b"{nope", headers={**auth_headers, "Content-Type": "application/json"})
    assert r.status_code == 422


def test_throttled_prepare_is_429_with_integer_retry_after(client, world, auth_headers, monkeypatch):
    async def throttled(*a, **k):
        raise vv.Throttled(120)

    monkeypatch.setattr(vv, "prepare", throttled)
    r = client.post(PREPARE, json={}, headers=auth_headers)
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "120"


@pytest.mark.parametrize("car_id", ["x" * 201, "a%00b"])
@pytest.mark.parametrize("suffix,method", [("/visual", "get"), ("/visual/prepare", "post")])
def test_bad_car_ids_are_422(client, world, auth_headers, car_id, suffix, method):
    assert getattr(client, method)(f"/api/v1/cars/{car_id}{suffix}", headers=auth_headers).status_code == 422


@pytest.mark.parametrize("method,url", [("get", VISUAL), ("post", PREPARE)])
def test_missing_visuals_tables_still_200_fallback(client, world, auth_headers, monkeypatch, method, url):
    async def missing(*a, **k):
        raise asyncpg.UndefinedTableError("relation does not exist")

    monkeypatch.setattr(crud, "get_visual", missing)
    r = getattr(client, method)(url, headers=auth_headers)
    assert r.status_code == 200
    assert (r.json()["state"], r.json()["reason"]) == ("fallback", "temporarily_unavailable")


# --- signed downloads -----------------------------------------------------------

def test_valid_download_serves_the_bytes(client, store):
    data = b"RIFF....WEBPfake"
    sha = _put(store, data)
    r = client.get(_signed_path(sha))  # no Authorization header
    assert "authorization" not in {k.lower() for k in r.request.headers}
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"
    assert r.headers["content-length"] == str(len(data))
    assert hashlib.sha256(r.content).hexdigest() == sha
    age = int(r.headers["cache-control"].split("max-age=")[1])
    assert r.headers["cache-control"].startswith("private") and 0 <= age <= get_settings().VISUAL_URL_TTL_SECONDS
    assert r.headers.get_list("x-content-type-options") == ["nosniff"]
    assert "content-security-policy" in r.headers  # the usual security headers still apply


def test_refused_downloads_are_never_cached(client, store):
    sha = _put(store)
    bad = [
        "/api/v1/visuals/files/not-a-real-name.webp",  # 404: bad name
        f"/api/v1/visuals/files/{sha}.webp?exp=1&sig={'0' * 64}",  # 404: unsigned
    ]
    for url in bad:
        r = client.get(url)
        assert r.status_code == 404
        cc = r.headers["cache-control"]
        assert cc.startswith("no-store") and "public" not in cc
    assert client.post(f"/api/v1/visuals/files/{sha}.webp").status_code == 405
    assert client.post(f"/api/v1/visuals/files/{sha}.webp").headers["cache-control"].startswith("no-store")
    ok = client.get(_signed_path(sha))  # the route's own header still wins on success
    assert ok.status_code == 200 and ok.headers["cache-control"].startswith("private, max-age=")
    assert "public" not in ok.headers["cache-control"] and "no-store" not in ok.headers["cache-control"]


def test_cache_age_is_clamped_to_remaining_lifetime(client, store):
    sha = _put(store)
    r = client.get(_signed_path(sha, now=time.time() - 800))  # about 100s of its 900s left
    assert r.status_code == 200
    assert 0 <= int(r.headers["cache-control"].split("max-age=")[1]) <= 100


def _bad_urls(sha):
    path, query = _signed_path(sha).split("?")
    exp, sig = [p.split("=")[1] for p in query.split("&")]
    return {
        "expired": _signed_path(sha, now=time.time() - 5000),
        "tampered": f"{path}?exp={exp}&sig={'0' * 64}",
        "short_sig": f"{path}?exp={exp}&sig={sig[:-1]}",
        "no_sig": f"{path}?exp={exp}",
        "no_exp": f"{path}?sig={sig}",
        "no_query": path,
        "non_hex": f"/api/v1/visuals/files/{'g' * 64}.webp?exp={exp}&sig={sig}",
        "upper": f"/api/v1/visuals/files/{sha.upper()}.webp?exp={exp}&sig={sig}",
        "traversal": f"/api/v1/visuals/files/..%2f..%2fetc%2fpasswd?exp={exp}&sig={sig}",
        "stream": f"/api/v1/visuals/files/{sha}.webp:stream?exp={exp}&sig={sig}",
        "long": f"/api/v1/visuals/files/{'a' * 70}?exp={exp}&sig={sig}",
        "unknown_sha": f"/api/v1/visuals/files/{'b' * 64}.webp?exp={exp}&sig={sig}",
    }


def test_every_refusal_is_the_same_404(client, store):
    sha = _put(store)
    for label, url in _bad_urls(sha).items():
        r = client.get(url)
        assert r.status_code == 404, label
        assert r.json()["message"] == GENERIC_404, label


def test_validly_signed_but_absent_object_is_the_same_404(client, store):
    r = client.get(_signed_path("c" * 64))
    assert r.status_code == 404 and r.json()["message"] == GENERIC_404


def test_object_whose_bytes_changed_is_404(client, store, caplog):
    sha = _put(store)
    store.object_path(sha).write_bytes(b"something else")
    r = client.get(_signed_path(sha))
    assert r.status_code == 404 and r.json()["message"] == GENERIC_404
    assert sha not in caplog.text


def test_oversized_object_is_404(client, store):
    sha = _put(store, b"x" * 409_601)
    assert client.get(_signed_path(sha)).status_code == 404


def test_download_is_rate_limited(client, store):
    url = _signed_path(_put(store))
    assert 429 in {client.get(url).status_code for _ in range(125)}


def test_download_does_not_touch_the_database(client, store, fake_pool):
    client.get(_signed_path(_put(store)))
    assert fake_pool.calls == []


def test_descriptor_url_round_trips_end_to_end(client, world, store, auth_headers):
    sha = _put(store, b"RIFF....WEBPfake-round-trip")
    world.rows[ECHO] = {"variant_key": ECHO, "state": "ready", "published_revision": 1, "sha256": sha}
    d = client.get(VISUAL, headers=auth_headers).json()
    assert d["state"] == "ready" and d["artwork"]["sha256"] == sha
    parts = urlsplit(d["artwork"]["download_url"])
    r = client.get(f"{parts.path}?{parts.query}")  # host stripped, no auth
    assert r.status_code == 200 and hashlib.sha256(r.content).hexdigest() == sha
