def test_app_imports_and_health_route_exists(client):
    paths = {r.path for r in client.app.routes}
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
