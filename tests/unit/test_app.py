def test_app_imports_and_health_route_exists(client):
    paths = {r.path for r in client.app.routes}
    assert "/health" in paths
    assert not any(p.startswith("/ws") or "metrics" in p for p in paths)


def test_no_header_logging(client, caplog):
    client.get("/health", headers={"Authorization": "Bearer secret-token-xyz"})
    assert "secret-token-xyz" not in caplog.text
