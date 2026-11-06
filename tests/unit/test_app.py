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
        return c.get("/health").json()


def test_health_pings_database_with_select_1(mocker, fake_pool):
    fake_pool.value = 1
    assert _health(mocker) == {"status": "ok", "database": True, "deepseek": True}
    assert ("fetchval", "select 1", ()) in fake_pool.calls


def test_health_database_down_keeps_status_ok(mocker, fake_pool):
    fake_pool.error = OSError("connection refused")
    assert _health(mocker) == {"status": "ok", "database": False, "deepseek": True}


def test_lifespan_opens_and_closes_the_pool(mocker, fake_pool):
    import main
    from fastapi.testclient import TestClient

    from app.db import pool

    init = mocker.patch("main.pool.init_pool", wraps=pool.init_pool)
    with TestClient(main.app):
        init.assert_awaited_once_with()
        assert not fake_pool.closed
    assert fake_pool.closed
