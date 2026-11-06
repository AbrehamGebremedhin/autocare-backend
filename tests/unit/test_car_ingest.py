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
    download = mocker.patch("app.CRUD.car_crud.FetchCarDataService.download_pdf")
    ingest = mocker.patch("app.CRUD.car_crud.kb.ingest_pdf", return_value=42)
    await ingest_car_manual(CAR)
    assert download.call_args.args[0] == "https://www.startmycar.com/files/echo.pdf"
    assert ingest.call_args.kwargs == {
        "source_type": "manual", "title": "Toyota Echo 2001 owner's manual", "car_id": "toyota-echo-2001",
    }
    assert update.call_args.args == ({"id": "toyota-echo-2001"}, {"kb_status": "ready", "kb_passage_count": 42})


async def test_ingest_zero_passages_is_no_manual(mocker, update):
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", return_value="/x.pdf")
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.download_pdf")
    mocker.patch("app.CRUD.car_crud.kb.ingest_pdf", return_value=0)
    await ingest_car_manual(CAR)
    assert update.call_args.args[1]["kb_status"] == "no_manual"


async def test_ingest_failure_sets_failed_logs_class_only_and_does_not_raise(mocker, update):
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.scrape_links", return_value="/x.pdf")
    mocker.patch("app.CRUD.car_crud.FetchCarDataService.download_pdf")
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


def test_delete_car_not_in_garage_is_404(client, auth_headers, fake_pool):
    fake_pool.status = "UPDATE 0"
    r = client.delete(f"/api/v1/cars/{CAR['id']}", headers=auth_headers)
    assert r.status_code == 404
