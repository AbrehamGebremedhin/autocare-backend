import pytest

PUBLIC = {"/health", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect"}


def test_every_non_auth_route_requires_token(client):
    for route in client.app.routes:
        path = getattr(route, "path", "")
        if path in PUBLIC or path.startswith("/api/v1/auth"):
            continue
        for method in route.methods - {"HEAD", "OPTIONS"}:
            r = client.request(method, path.replace("{", "").replace("}", ""))
            assert r.status_code == 401, (method, path)


def test_debug_auth_endpoints_gone(client):
    for p in ["/api/v1/auth/list-users", "/api/v1/auth/manual-confirm", "/api/v1/auth/debug-confirm"]:
        assert client.get(p).status_code in (404, 405)


def test_jwt_uses_settings_secret(monkeypatch):
    # a token signed with the old hard-coded fallback secret is rejected
    import jwt
    from app.core.config import get_settings
    from app.utils.auth_middleware import jwt_handler, AuthenticationError
    assert jwt_handler.secret_key == get_settings().JWT_SECRET_KEY
    bad = jwt.encode({"sub": "u", "type": "access"}, "your-super-secret-jwt-key-change-in-production", "HS256")
    with pytest.raises(AuthenticationError):
        jwt_handler.verify_token(bad)
