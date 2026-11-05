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


def test_confirm_works_and_logs_no_secrets(client, caplog):
    import logging
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.db.base import get_db_handler
    from app.api.v1 import user_auth

    token, email = "SECRET-TOKEN-abc123", "victim@example.com"
    db = MagicMock()
    db.auth.verify_otp.return_value = SimpleNamespace(user=SimpleNamespace(id="user-uuid-1"))
    handler = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=db)
    cm.__aexit__ = AsyncMock(return_value=False)
    handler.get_connection.return_value = cm
    client.app.dependency_overrides[get_db_handler] = lambda: handler
    try:
        with caplog.at_level(logging.DEBUG), patch.object(
            user_auth.user_service, "ensure_user_profile", AsyncMock()
        ):
            r = client.post("/api/v1/auth/confirm", json={"token": token, "email": email})
    finally:
        client.app.dependency_overrides.pop(get_db_handler, None)
    assert r.status_code == 200
    out = caplog.text + r.text
    assert token not in out and email not in out and "user-uuid-1" not in out


def _db_override(client, db):
    from unittest.mock import AsyncMock, MagicMock
    from app.db.base import get_db_handler

    handler = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=db)
    cm.__aexit__ = AsyncMock(return_value=False)
    handler.get_connection.return_value = cm
    client.app.dependency_overrides[get_db_handler] = lambda: handler


def test_register_and_login_errors_log_only_exception_class(client, caplog):
    import logging
    from unittest.mock import MagicMock
    from app.db.base import get_db_handler

    secret = "upstream said: password=hunter2"
    db = MagicMock()
    db.auth.sign_up.side_effect = RuntimeError(secret)
    db.auth.sign_in_with_password.side_effect = RuntimeError(secret)
    _db_override(client, db)
    try:
        with caplog.at_level(logging.DEBUG):
            r1 = client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "pw123456"})
            r2 = client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "pw123456"})
    finally:
        client.app.dependency_overrides.pop(get_db_handler, None)
    assert r1.status_code >= 400 and r2.status_code >= 400
    assert "RuntimeError" in caplog.text
    assert secret not in caplog.text + r1.text + r2.text


def test_confirm_failure_detail_is_generic(client, caplog):
    import logging
    from unittest.mock import MagicMock
    from app.db.base import get_db_handler

    secret = "db host 10.0.0.5 refused"
    db = MagicMock()
    db.auth.verify_otp.side_effect = RuntimeError(secret)
    db.auth.exchange_code_for_session.side_effect = RuntimeError("other")
    _db_override(client, db)
    try:
        with caplog.at_level(logging.DEBUG):
            r = client.post("/api/v1/auth/confirm", json={"token": "tok-xyz"})
    finally:
        client.app.dependency_overrides.pop(get_db_handler, None)
    assert r.status_code == 400
    assert secret not in r.text + caplog.text
    assert "tok-xyz" not in r.text + caplog.text


def test_logout_requires_bearer_header_not_query(client):
    r = client.post("/api/v1/auth/logout", params={"token": "abc"})
    assert r.status_code == 401


def test_logout_validates_token_and_never_calls_supabase(client, auth_headers, caplog, monkeypatch):
    import logging
    from unittest.mock import MagicMock
    from app.db import base
    from app.utils import auth_middleware

    token = auth_headers["Authorization"].split(" ", 1)[1]
    fake = MagicMock()
    monkeypatch.setattr(base, "create_client", lambda *a, **k: fake)
    with caplog.at_level(logging.DEBUG):
        r = client.post("/api/v1/auth/logout", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == {"message": "Logged out successfully."}
    auth_middleware.validate_user_with_supabase.assert_awaited_once()
    fake.auth.admin.sign_out.assert_not_called()
    assert token not in caplog.text
    assert "Logout successful" in caplog.text


def test_logout_with_invalid_token_is_401(client):
    r = client.post("/api/v1/auth/logout", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401


def _real_connection_auth(monkeypatch, **side_effects):
    """Route requests through the real get_connection with a fake Supabase client (the 62ff578 regression path)."""
    from unittest.mock import MagicMock
    from app.db import base

    fake = MagicMock()
    for name, effect in side_effects.items():
        getattr(fake.auth, name).side_effect = effect
    monkeypatch.setattr(base, "create_client", lambda *a, **k: fake)
    return fake


@pytest.mark.parametrize(
    "error, status",
    [
        (("Invalid login credentials", 400, "invalid_credentials"), 401),
        (("Email not confirmed", 400, "email_not_confirmed"), 401),
        (("Invalid login credentials", 400, None), 401),  # older GoTrue sends no code
        (("Request rate limit reached", 429, "over_request_rate_limit"), 429),
        (("upstream exploded", 500, "unexpected_failure"), 500),
    ],
)
def test_login_maps_supabase_auth_errors(client, monkeypatch, error, status):
    from supabase import AuthApiError

    _real_connection_auth(monkeypatch, sign_in_with_password=AuthApiError(*error))
    r = client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "pw123456"})
    assert r.status_code == status
    assert error[0] not in r.text


def test_wrong_password_is_401_with_generic_message(client, monkeypatch):
    from supabase import AuthApiError

    _real_connection_auth(
        monkeypatch, sign_in_with_password=AuthApiError("Invalid login credentials", 400, "invalid_credentials")
    )
    r = client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "wrong-pass"})
    assert r.status_code == 401
    assert r.json()["message"] == "Invalid email or password"


@pytest.mark.parametrize(
    "error, status",
    [
        (("User already registered", 422, "user_already_exists"), 409),
        (("A user with this email address has already been registered", 422, "email_exists"), 409),
        (("Unable to validate email address: invalid format", 400, "email_address_invalid"), 400),
        (("Signups not allowed for this instance", 422, "signup_disabled"), 400),
        (("Email rate limit exceeded", 429, "over_email_send_rate_limit"), 429),
    ],
)
def test_register_maps_supabase_auth_errors(client, monkeypatch, error, status):
    from supabase import AuthApiError

    _real_connection_auth(monkeypatch, sign_up=AuthApiError(*error))
    r = client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "pw123456"})
    assert r.status_code == status
    assert error[0] not in r.text


def test_register_weak_password_is_400(client, monkeypatch):
    from supabase import AuthWeakPasswordError

    _real_connection_auth(
        monkeypatch, sign_up=AuthWeakPasswordError("Password should be at least 6 characters", 422, ["length"])
    )
    r = client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "pw"})
    assert r.status_code == 400
    assert r.json()["message"] == "Password does not meet security requirements"


def test_non_auth_failure_in_connection_block_stays_generic_500(client, monkeypatch):
    _real_connection_auth(monkeypatch, sign_in_with_password=RuntimeError("socket to 10.0.0.5 reset"))
    r = client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "pw123456"})
    assert r.status_code == 500
    assert "10.0.0.5" not in r.text
