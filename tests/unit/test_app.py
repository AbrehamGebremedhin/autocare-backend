import asyncio
import logging

import pytest
from fastapi.routing import iter_route_contexts

from app.utils.logger import get_logger_instance


def test_app_imports_and_health_route_exists(client):
    paths = {r.path for r in iter_route_contexts(client.app.routes)}
    assert "/health" in paths
    assert not any(p.startswith("/ws") or "metrics" in p for p in paths)


def test_no_header_logging(client, caplog):
    client.get("/health", headers={"Authorization": "Bearer secret-token-xyz"})
    assert "secret-token-xyz" not in caplog.text


def test_health_reports_dependencies(client):
    assert client.get("/health").json() == {"status": "ok", "database": True, "deepseek": True}


def test_health_reports_deepseek_down(client, mocker):
    mocker.patch("main.llm.ping", return_value=False)
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["deepseek"] is False


def _health(mocker):
    import main
    from fastapi.testclient import TestClient

    mocker.patch("main.llm.ping", return_value=True)
    with TestClient(main.app) as c:
        return c.get("/health")


def test_health_pings_database_with_select_1(mocker, fake_pool):
    fake_pool.value = 1
    r = _health(mocker)
    assert r.status_code == 200 and r.json() == {"status": "ok", "database": True, "deepseek": True}
    assert ("fetchval", "select 1", ()) in fake_pool.calls


def test_health_database_down_is_503(mocker, fake_pool):
    # The compose healthcheck then marks the api unhealthy; a DeepSeek outage alone keeps 200.
    fake_pool.error = OSError("connection refused")
    r = _health(mocker)
    assert r.status_code == 503
    assert r.json() == {"status": "unavailable", "database": False, "deepseek": True}


def test_lifespan_opens_and_closes_the_pool(mocker, fake_pool):
    import main
    from fastapi.testclient import TestClient

    from app.db import pool

    init = mocker.patch("main.pool.init_pool", wraps=pool.init_pool)
    with TestClient(main.app):
        init.assert_awaited_once_with()
        assert not fake_pool.closed
    assert fake_pool.closed


def test_media_dir_writable_checks_nearest_existing_folder(tmp_path):
    import main

    assert main.media_dir_writable(str(tmp_path))
    assert main.media_dir_writable(str(tmp_path / "not" / "yet"))  # uploads create missing folders
    (tmp_path / "file").write_text("x")
    assert not main.media_dir_writable(str(tmp_path / "file"))
    assert not main.media_dir_writable(str(tmp_path / "file" / "below"))


def _startup_log(mocker, monkeypatch, caplog, media_dir: str, writable: bool) -> str:
    import logging

    import main
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main.settings, "MEDIA_DIR", media_dir)
    mocker.patch("main.os.access", return_value=writable)
    with caplog.at_level(logging.WARNING, logger="autocare"), TestClient(main.app):
        pass
    return caplog.text


def test_startup_warns_when_media_dir_not_writable(mocker, monkeypatch, caplog, tmp_path, fake_pool):
    text = _startup_log(mocker, monkeypatch, caplog, str(tmp_path), writable=False)
    assert "MEDIA_DIR" in text and "not writable" in text


def test_startup_quiet_when_media_dir_writable(mocker, monkeypatch, caplog, tmp_path, fake_pool):
    assert "MEDIA_DIR" not in _startup_log(mocker, monkeypatch, caplog, str(tmp_path), writable=True)


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_docs_are_off_by_default(client, path):
    r = client.get(path)
    assert r.status_code == 404
    assert r.json()["error_code"] == "SYS_004"


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_docs_are_served_when_enabled(path):
    import main
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    assert TestClient(FastAPI(**main.docs_urls(True))).get(path).status_code == 200
    assert TestClient(FastAPI(**main.docs_urls(False))).get(path).status_code == 404


def test_cors_exposes_retry_after_to_browsers(client):
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    exposed = r.headers["access-control-expose-headers"]
    assert "Retry-After" in exposed and "X-Correlation-ID" in exposed


async def test_logging_never_waits_for_the_thread_pool(monkeypatch, caplog):
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "run_in_executor", lambda *a, **k: pytest.fail("logging used the executor"))
    with caplog.at_level(logging.INFO):
        await get_logger_instance("pool-test").info("hello")
    assert "hello" in caplog.text


async def test_lifespan_installs_a_bigger_default_executor(mocker):
    import main

    mocker.patch("main.pool.init_pool")
    mocker.patch("main.pool.close_pool")
    async with main.lifespan(main.app):
        executor = asyncio.get_running_loop()._default_executor
    assert executor._max_workers == main.DEFAULT_EXECUTOR_THREADS == 32


def test_startup_removes_stale_turn_and_manual_directories(monkeypatch, tmp_path):
    import tempfile

    import main

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for name in ("autocare-turn-a", "autocare-manual-b", "other-c"):
        (tmp_path / name).mkdir()
    assert main.sweep_stale_tmp() == 2
    assert [p.name for p in tmp_path.iterdir()] == ["other-c"]


def _wait_for(condition, seconds=3.0):
    import time

    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        time.sleep(0.02)
    return condition()


def test_models_are_not_loaded_at_startup_by_default(mocker, fake_pool):
    import main
    from fastapi.testclient import TestClient

    prefetch = mocker.patch("main.hearing.prefetch")
    with TestClient(main.app):
        pass
    prefetch.assert_not_called()


def test_models_are_loaded_at_startup_when_enabled(mocker, monkeypatch, fake_pool):
    import main
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main.settings, "PRELOAD_MODELS", True)
    prefetch = mocker.patch("main.hearing.prefetch")
    with TestClient(main.app):
        assert _wait_for(lambda: prefetch.called)


def test_a_failed_preload_is_logged_and_the_app_still_starts(mocker, monkeypatch, fake_pool, caplog):
    import main
    from fastapi.testclient import TestClient

    fake_pool.value = 1
    monkeypatch.setattr(main.settings, "PRELOAD_MODELS", True)
    mocker.patch("main.llm.ping", return_value=True)
    mocker.patch("main.hearing.prefetch", side_effect=OSError("no disk /secret/path"))
    with caplog.at_level(logging.WARNING, logger="autocare"), TestClient(main.app) as c:
        assert _wait_for(lambda: "OSError" in caplog.text)
        assert c.get("/health").status_code == 200
    assert "/secret/path" not in caplog.text


def test_startup_resumes_interrupted_ingests(mocker, fake_pool):
    import main
    from fastapi.testclient import TestClient

    resume = mocker.patch("main.resume_pending_ingests", mocker.AsyncMock())
    with TestClient(main.app):
        assert _wait_for(lambda: resume.await_count == 1)
