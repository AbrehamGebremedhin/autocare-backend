def test_security_headers_on_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    h = response.headers
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["X-XSS-Protection"] == "1; mode=block"
    assert h["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert h["Permissions-Policy"] == "geolocation=(), microphone=(), camera=()"
    assert h["X-Frame-Options"] == "DENY"
    assert h["Cache-Control"] == "public, max-age=300"
    assert "frame-ancestors 'none'" in h["Content-Security-Policy"]
    assert "Strict-Transport-Security" not in h  # only set when ENV=production


def test_csp_nonce_header_matches_csp(client):
    response = client.get("/health")

    nonce = response.headers["X-CSP-Nonce"]
    assert nonce
    assert f"'nonce-{nonce}'" in response.headers["Content-Security-Policy"]


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
