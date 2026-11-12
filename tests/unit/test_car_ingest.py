import asyncio
from pathlib import Path

import asyncpg
import pytest

from app.CRUD.car_crud import ingest_car_manual
from scripts.ingest_kb import parse_pdf_name

CAR = {"id": "toyota-echo-2001", "make": "Toyota", "model": "Echo", "year": 2001}
NEW_CAR = {"make": "Toyota", "model": "Echo", "year": 2001}


def _stub_add_car(mocker, car_info):
    mocker.patch(
        "app.api.v1.car_route.user_profile_service.add_car_to_user",
        return_value={"success": True, "car_info": car_info},
    )


def test_add_car_returns_before_ingest(client, auth_headers, mocker):
    ingest = mocker.patch("app.api.v1.car_route.ingest_car_manual")
    _stub_add_car(mocker, {**CAR, "kb_status": "pending"})
    r = client.post("/api/v1/cars", json=NEW_CAR, headers=auth_headers)
    assert r.status_code == 200 and r.json()["kb_status"] == "pending"
    ingest.assert_called_once_with({**CAR, "kb_status": "pending"})


def test_add_car_defaults_to_pending_when_row_has_no_status(client, auth_headers, mocker):
    ingest = mocker.patch("app.api.v1.car_route.ingest_car_manual")
    _stub_add_car(mocker, CAR)
    r = client.post("/api/v1/cars", json=NEW_CAR, headers=auth_headers)
    assert r.json()["kb_status"] == "pending" and r.json()["kb_passage_count"] == 0
    ingest.assert_called_once()


def test_add_car_skips_ingest_when_already_ready(client, auth_headers, mocker):
    ingest = mocker.patch("app.api.v1.car_route.ingest_car_manual")
    _stub_add_car(mocker, {**CAR, "kb_status": "ready", "kb_passage_count": 9})
    r = client.post("/api/v1/cars", json=NEW_CAR, headers=auth_headers)
    assert r.json()["kb_status"] == "ready"
    ingest.assert_not_called()


@pytest.fixture
def update(mocker):
    return mocker.patch("app.CRUD.car_crud.CarCRUD.update")


async def test_ingest_marks_no_manual(mocker, update):
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", return_value=None)
    await ingest_car_manual(CAR)
    assert update.call_args.args[1]["kb_status"] == "no_manual"


async def test_ingest_ready_path(mocker, update):
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", return_value="/files/echo.pdf")
    download = mocker.patch("app.CRUD.car_crud.manual_search.fetch_pdf", return_value=True)
    ingest = mocker.patch("app.CRUD.car_crud.kb.ingest_pdf", return_value=42)
    await ingest_car_manual(CAR)
    assert download.call_args.args[0] == "https://www.startmycar.com/files/echo.pdf"
    assert ingest.call_args.kwargs == {
        "source_type": "manual", "title": "Toyota Echo 2001 owner's manual", "car_id": "toyota-echo-2001",
    }
    assert update.call_args.args == ({"id": "toyota-echo-2001"}, {"kb_status": "ready", "kb_passage_count": 42})


async def test_startmycar_manual_that_fails_the_fetch_is_no_manual(mocker, update):
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", return_value="/files/echo.pdf")
    mocker.patch("app.CRUD.car_crud.manual_search.fetch_pdf", return_value=False)
    ingest = mocker.patch("app.CRUD.car_crud.kb.ingest_pdf")
    await ingest_car_manual(CAR)
    ingest.assert_not_called()
    assert update.call_args.args[1]["kb_status"] == "no_manual"


async def test_ingest_zero_passages_is_no_manual(mocker, update):
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", return_value="/x.pdf")
    mocker.patch("app.CRUD.car_crud.manual_search.fetch_pdf", return_value=True)
    mocker.patch("app.CRUD.car_crud.kb.ingest_pdf", return_value=0)
    await ingest_car_manual(CAR)
    assert update.call_args.args[1]["kb_status"] == "no_manual"


async def test_ingest_failure_sets_failed_logs_class_only_and_does_not_raise(mocker, update):
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", return_value="/x.pdf")
    mocker.patch("app.CRUD.car_crud.manual_search.fetch_pdf", return_value=True)
    mocker.patch("app.CRUD.car_crud.kb.ingest_pdf", side_effect=RuntimeError("secret-token-123"))
    log_error = mocker.patch("app.CRUD.car_crud._logger.error")
    await ingest_car_manual(CAR)
    assert update.call_args.args[1]["kb_status"] == "failed"
    logged = " ".join(str(c.args[0]) for c in log_error.call_args_list)
    assert "RuntimeError" in logged and "secret-token-123" not in logged


async def test_ingest_survives_status_update_failure(mocker):
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", side_effect=ValueError("boom"))
    mocker.patch("app.CRUD.car_crud.CarCRUD.update", side_effect=ConnectionError("db down"))
    await ingest_car_manual(CAR)


def test_book_title_from_filename():
    assert parse_pdf_name(Path("car_data/haynes-small-engine-repair-manual.pdf")) == {
        "source_type": "book", "title": "Haynes Small Engine Repair Manual", "car_id": None,
    }


def test_manual_filename_maps_to_car_id():
    parsed = parse_pdf_name(Path("car_data/toyota-echo-2001_manual.pdf"))
    assert parsed["source_type"] == "manual" and parsed["car_id"] == "toyota-echo-2001"


def test_script_does_not_run_on_import():
    import scripts.ingest_kb as mod

    assert callable(mod.main)


def test_add_car_database_error_is_generic_500(client, auth_headers, mocker):
    ingest = mocker.patch("app.api.v1.car_route.ingest_car_manual")
    mocker.patch(
        "app.api.v1.car_route.user_profile_service.car_crud.create_or_get_car",
        side_effect=RuntimeError("pg: password=hunter2"),
    )
    r = client.post("/api/v1/cars", json=NEW_CAR, headers=auth_headers)
    assert r.status_code == 500
    assert "hunter2" not in r.text and r.json()["message"] == "Could not add car"
    ingest.assert_not_called()


def test_add_car_fails_when_the_array_append_fails(client, auth_headers, mocker, fake_pool):
    ingest = mocker.patch("app.api.v1.car_route.ingest_car_manual")
    mocker.patch(
        "app.api.v1.car_route.user_profile_service.car_crud.create_or_get_car",
        return_value=[{**CAR, "kb_status": "pending"}],
    )
    real_execute = fake_pool.execute

    async def execute(sql, *args):
        if "array_append" in sql:
            raise RuntimeError("down")
        return await real_execute(sql, *args)

    fake_pool.execute = execute
    r = client.post("/api/v1/cars", json=NEW_CAR, headers=auth_headers)
    assert r.status_code == 500
    ingest.assert_not_called()


async def test_create_car_inserts_on_conflict_do_nothing_then_reads_back(fake_pool):
    from app.CRUD.car_crud import CarCRUD

    row = {**CAR, "kb_status": "pending", "car_guide_links": []}
    fake_pool.rows = []  # first read: not there yet
    original = fake_pool.fetch

    async def fetch(sql, *args):
        fake_pool.rows = [row] if any(c[0] == "execute" for c in fake_pool.calls) else []
        return await original(sql, *args)

    fake_pool.fetch = fetch
    cars = await CarCRUD().create_or_get_car(dict(NEW_CAR))
    assert cars == [row]
    [(m1, _, _), (m2, sql, args), (m3, _, _)] = fake_pool.calls
    assert (m1, m2, m3) == ("fetch", "execute", "fetch")
    assert sql.startswith('INSERT INTO "Car" (') and sql.endswith('ON CONFLICT ("id") DO NOTHING')
    assert "toyota-echo-2001" in args and "Toyota" in args and "toyota-echo-2001" not in sql


async def test_concurrent_adds_of_the_same_new_car_do_not_raise(fake_pool):
    from app.CRUD.car_crud import CarCRUD

    row = {**CAR, "kb_status": "pending", "car_guide_links": []}
    fake_pool.rows = []
    original = fake_pool.fetch

    async def fetch(sql, *args):
        # Both callers see no row on their first read; once either insert has run, the row exists.
        fake_pool.rows = [row] if any(c[0] == "execute" for c in fake_pool.calls) else []
        return await original(sql, *args)

    fake_pool.fetch = fetch
    real_execute = fake_pool.execute

    async def execute(sql, *args):
        # What the real table does: a plain INSERT that finds the row already there raises.
        if "ON CONFLICT" not in sql and sql.startswith("INSERT") and fake_pool.rows:
            raise asyncpg.UniqueViolationError("duplicate key")
        return await real_execute(sql, *args)

    fake_pool.execute = execute
    first, second = await asyncio.gather(CarCRUD().create_or_get_car(dict(NEW_CAR)), CarCRUD().create_or_get_car(dict(NEW_CAR)))
    assert first == second == [row]


def test_delete_car_database_error_is_500_not_404(client, auth_headers, fake_pool):
    fake_pool.error = RuntimeError("connection reset")
    r = client.delete(f"/api/v1/cars/{CAR['id']}", headers=auth_headers)
    assert r.status_code == 500
    assert "connection reset" not in r.text and r.json()["message"] == "Could not remove car"


@pytest.mark.parametrize(
    "body",
    [
        {**NEW_CAR, "year": 99999999999},
        {**NEW_CAR, "year": 1899},
        {**NEW_CAR, "year": __import__("datetime").date.today().year + 2},
        {**NEW_CAR, "make": "Toy\x00ota"},
        {**NEW_CAR, "model": "Ec\x00ho"},
        {**NEW_CAR, "make": ""},
        {**NEW_CAR, "model": "x" * 65},
    ],
)
def test_add_car_rejects_bad_input_with_422_before_the_database(client, auth_headers, mocker, body):
    add = mocker.patch("app.api.v1.car_route.user_profile_service.add_car_to_user")
    r = client.post("/api/v1/cars", json=body, headers=auth_headers)
    assert r.status_code == 422
    add.assert_not_called()


def test_add_car_accepts_next_years_model(client, auth_headers, mocker):
    import datetime

    mocker.patch("app.api.v1.car_route.ingest_car_manual")
    _stub_add_car(mocker, CAR)
    body = {**NEW_CAR, "year": datetime.date.today().year + 1}
    assert client.post("/api/v1/cars", json=body, headers=auth_headers).status_code == 200


@pytest.mark.parametrize("car_id", ["toyota%00echo", "x" * 201])  # %00 arrives as NUL
def test_delete_car_rejects_bad_ids_with_422(client, auth_headers, mocker, car_id):
    remove = mocker.patch("app.api.v1.car_route.user_profile_service.remove_car_from_user")
    r = client.delete(f"/api/v1/cars/{car_id}", headers=auth_headers)
    assert r.status_code == 422
    remove.assert_not_called()


def test_delete_car_not_in_garage_is_404(client, auth_headers, fake_pool):
    fake_pool.status = "UPDATE 0"
    r = client.delete(f"/api/v1/cars/{CAR['id']}", headers=auth_headers)
    assert r.status_code == 404


def test_scrape_client_does_not_advertise_brotli_without_a_decoder():
    import importlib.util

    from app.services.fetch_car_data_service import FetchCarDataService

    async def build():
        service = FetchCarDataService()
        try:
            return (await service._get_client()).headers["accept-encoding"]
        finally:
            await service.cleanup()

    offered = {e.strip() for e in asyncio.run(build()).split(",")}
    has_brotli = any(importlib.util.find_spec(m) for m in ("brotli", "brotlicffi"))
    assert has_brotli or "br" not in offered


# --- limits on adding cars ---------------------------------------------------------

def _post_car(client, headers, **extra):
    return client.post("/api/v1/cars", json={**NEW_CAR, **extra}, headers=headers)


@pytest.fixture
def add_car(mocker):
    """add_car_to_user and the background ingest stubbed; returns the add_car_to_user mock."""
    mocker.patch("app.api.v1.car_route.ingest_car_manual")
    return mocker.patch(
        "app.api.v1.car_route.user_profile_service.add_car_to_user",
        return_value={"success": True, "car_info": {**CAR, "kb_status": "ready"}},
    )


def test_add_car_is_rate_limited_per_address(client, auth_headers, add_car):
    assert [_post_car(client, auth_headers).status_code for _ in range(10)] == [200] * 10
    r = _post_car(client, auth_headers)
    assert r.status_code == 429 and r.json()["error_code"] == "EXT_003"


def test_full_garage_refuses_a_new_car(client, auth_headers, add_car, mocker, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "MAX_CARS_PER_USER", 2)
    mocker.patch("app.api.v1.car_route.user_profile_service.garage_ids", return_value=["a-b-2001", "c-d-2002"])
    r = _post_car(client, auth_headers)
    assert r.status_code == 422 and r.json()["error_code"] == "BIZ_001"
    assert r.json()["message"] == "A garage holds at most 2 cars. Remove one to add another."
    add_car.assert_not_called()


def test_full_garage_still_returns_a_car_already_in_it(client, auth_headers, add_car, mocker, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "MAX_CARS_PER_USER", 2)
    mocker.patch("app.api.v1.car_route.user_profile_service.garage_ids", return_value=["toyota-echo-2001", "c-d-2002"])
    assert _post_car(client, auth_headers).status_code == 200
    add_car.assert_called_once()


def test_daily_car_add_limit_is_429_with_retry_after(client, auth_headers, add_car, monkeypatch):
    from datetime import datetime, timezone

    from app.core.config import get_settings
    from app.services import usage_limits

    monkeypatch.setattr(get_settings(), "DAILY_CAR_ADD_LIMIT", 1)
    monkeypatch.setattr(usage_limits, "_utc_now", lambda: datetime(2026, 3, 1, 23, 0, 0, tzinfo=timezone.utc))
    assert _post_car(client, auth_headers).status_code == 200
    r = _post_car(client, auth_headers)
    assert r.status_code == 429 and r.headers["retry-after"] == "3600"
    assert r.json()["message"] == "Daily limit of 1 added cars reached. Try again after midnight UTC."
    assert add_car.call_count == 1
    assert usage_limits._counts == {}  # its own counter, not the message one


def test_daily_car_add_limit_zero_turns_it_off(client, auth_headers, add_car, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "DAILY_CAR_ADD_LIMIT", 0)
    monkeypatch.setattr(get_settings(), "MAX_CARS_PER_USER", 0)
    assert [_post_car(client, auth_headers).status_code for _ in range(5)] == [200] * 5


async def test_second_ingest_of_the_same_car_is_a_no_op_while_the_first_runs(mocker, update):
    from app.CRUD import car_crud

    started, release, calls = asyncio.Event(), asyncio.Event(), []

    async def scrape(*args, **kwargs):
        calls.append(1)
        started.set()
        await release.wait()
        return None

    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", side_effect=scrape)
    first = asyncio.create_task(ingest_car_manual(CAR))
    await started.wait()
    await asyncio.wait_for(ingest_car_manual(CAR), 1)  # returns at once
    assert len(calls) == 1 and CAR["id"] in car_crud._ingesting
    release.set()
    await first
    assert CAR["id"] not in car_crud._ingesting


# --- resuming interrupted ingests ----------------------------------------------------

async def test_resume_marks_pending_cars_failed_then_ingests_each(fake_pool, mocker):
    from app.CRUD import car_crud

    fake_pool.rows = [{**CAR, "kb_status": "failed"}, {**CAR, "id": "honda-fit-2009", "kb_status": "failed"}]
    ingest = mocker.patch("app.CRUD.car_crud.ingest_car_manual")
    await car_crud.resume_pending_ingests()
    [(method, sql, _)] = fake_pool.calls
    assert sql == "UPDATE \"Car\" SET kb_status = 'failed' WHERE kb_status = 'pending' RETURNING *"
    assert [c.args[0]["id"] for c in ingest.await_args_list] == ["toyota-echo-2001", "honda-fit-2009"]


async def test_resume_survives_a_database_error(fake_pool, mocker):
    from app.CRUD import car_crud

    fake_pool.error = RuntimeError("db down secret")
    log_error = mocker.patch("app.CRUD.car_crud._logger.error")
    assert await car_crud.resume_pending_ingests() is None
    logged = " ".join(str(c.args[0]) for c in log_error.call_args_list)
    assert "RuntimeError" in logged and "secret" not in logged


async def test_resume_continues_after_one_ingest_raises(fake_pool, mocker):
    from app.CRUD import car_crud

    fake_pool.rows = [CAR, {**CAR, "id": "honda-fit-2009"}]
    ingest = mocker.patch("app.CRUD.car_crud.ingest_car_manual", side_effect=[RuntimeError("boom"), None])
    await car_crud.resume_pending_ingests()
    assert ingest.await_count == 2
