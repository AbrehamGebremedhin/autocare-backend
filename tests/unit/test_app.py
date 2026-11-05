def test_app_imports_and_health_route_exists(client):
    paths = {r.path for r in client.app.routes}
    assert "/health" in paths
    assert not any(p.startswith("/ws") or "metrics" in p for p in paths)


def test_no_header_logging(client, caplog):
    client.get("/health", headers={"Authorization": "Bearer secret-token-xyz"})
    assert "secret-token-xyz" not in caplog.text


def test_health_reports_dependencies(client):
    assert client.get("/health").json() == {"status": "ok", "supabase": True, "deepseek": True}


def test_health_reports_deepseek_down(client, mocker):
    mocker.patch("main.llm.ping", return_value=False)
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["deepseek"] is False


def test_health_supabase_down_keeps_status_ok(mocker):
    import main
    from fastapi.testclient import TestClient

    mocker.patch("main.llm.ping", return_value=True)
    mocker.patch("main._ping_supabase", side_effect=RuntimeError("down"))
    with TestClient(main.app) as c:
        body = c.get("/health").json()
    assert body == {"status": "ok", "supabase": False, "deepseek": True}
