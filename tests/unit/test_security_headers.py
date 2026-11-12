def test_security_headers_on_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    h = response.headers
    assert h["X-Content-Type-Options"] == "nosniff"
    assert "X-XSS-Protection" not in h
    assert h["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert h["Permissions-Policy"] == "geolocation=(), microphone=(), camera=()"
    assert h["X-Frame-Options"] == "DENY"
    assert h["Cache-Control"] == "no-store"  # nothing is cacheable unless the route says so
    assert "frame-ancestors 'none'" in h["Content-Security-Policy"]
    assert "Strict-Transport-Security" not in h  # only set when ENV=production


def test_no_nonce_header_and_a_locked_down_csp(client):
    h = client.get("/health").headers
    assert "X-CSP-Nonce" not in h
    assert h["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"


def test_cars_responses_are_not_cached(client, auth_headers, mocker):
    mocker.patch("app.api.v1.car_route.user_profile_service.get_user_cars", return_value=[])
    response = client.get("/api/v1/cars", headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["Cache-Control"].startswith("no-store")


def test_csp_has_no_unsafe_inline_or_websockets(client):
    csp = client.get("/health").headers["Content-Security-Policy"]
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "ws:" not in csp and "wss:" not in csp


def test_docs_pages_skip_csp_only_when_docs_are_on(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.utils.security_middleware import SecurityHeadersMiddleware

    def docs_app(enabled):
        monkeypatch.setattr(get_settings(), "ENABLE_DOCS", enabled)
        app = FastAPI(**({} if enabled else {"docs_url": None, "redoc_url": None, "openapi_url": None}))
        app.add_middleware(SecurityHeadersMiddleware)
        return TestClient(app)

    on = docs_app(True)
    for path in ("/docs", "/redoc"):
        assert on.get(path).status_code == 200 and "Content-Security-Policy" not in on.get(path).headers
    # a path that merely contains "/docs" is not exempt
    assert "Content-Security-Policy" in on.get("/api/v1/cars/docs").headers

    off = docs_app(False)  # the 404s at those paths get the full headers
    for path in ("/docs", "/redoc"):
        r = off.get(path)
        assert r.status_code == 404
        assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"] and r.headers["X-Frame-Options"] == "DENY"


def _bare_app(monkeypatch, **settings):
    from fastapi import FastAPI, Response
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.utils.security_middleware import SecurityHeadersMiddleware

    for name, value in settings.items():
        monkeypatch.setattr(get_settings(), name, value)
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/own")
    def own():
        return Response("x", headers={"Cache-Control": "private, max-age=60"})

    return TestClient(app)


def test_a_routes_own_cache_control_wins(monkeypatch):
    assert _bare_app(monkeypatch).get("/own").headers["Cache-Control"] == "private, max-age=60"


def test_hsts_follows_the_settings_not_the_environment(monkeypatch):
    monkeypatch.delenv("ENV", raising=False)
    assert "Strict-Transport-Security" not in _bare_app(monkeypatch, ENV="development").get("/own").headers
    r = _bare_app(monkeypatch, ENV="production").get("/own")
    assert r.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"


def test_streaming_bodies_pass_through_the_headers_middleware(monkeypatch):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient

    from app.utils.security_middleware import SecurityHeadersMiddleware

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/stream")
    def stream():
        return StreamingResponse(iter([b"a", b"b"]), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    r = TestClient(app).get("/stream")
    assert r.text == "ab" and r.headers["Cache-Control"] == "no-cache" and r.headers["X-Content-Type-Options"] == "nosniff"


def test_production_refuses_debug():
    import pytest
    from pydantic import ValidationError

    from app.core.config import Settings

    base = {"JWT_SECRET_KEY": "k" * 32, "DATABASE_URL": "postgresql://a@b/c", "DEEPSEEK_API_KEY": "k", "_env_file": None}
    with pytest.raises(ValidationError, match="DEBUG"):
        Settings(ENV="production", DEBUG=True, **base)
    assert Settings(ENV="production", DEBUG=False, **base).is_production()
    assert Settings(ENV="development", DEBUG=True, **base).DEBUG
