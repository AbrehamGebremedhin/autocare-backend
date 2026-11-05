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


def test_logout_uses_header_token_and_never_logs_it(client, auth_headers, caplog):
    import logging
    from unittest.mock import MagicMock
    from app.db.base import get_db_handler

    token = auth_headers["Authorization"].split(" ", 1)[1]
    db = MagicMock()
    _db_override(client, db)
    try:
        with caplog.at_level(logging.DEBUG):
            r = client.post("/api/v1/auth/logout", headers=auth_headers)
    finally:
        client.app.dependency_overrides.pop(get_db_handler, None)
    assert r.status_code == 200
    db.auth.admin.sign_out.assert_called_once_with(token)
    assert token not in caplog.text
    assert "Logout successful" in caplog.text


def test_logout_failure_is_generic_400(client, auth_headers, caplog):
    import logging
    from unittest.mock import MagicMock
    from app.db.base import get_db_handler

    token = auth_headers["Authorization"].split(" ", 1)[1]
    db = MagicMock()
    db.auth.admin.sign_out.side_effect = RuntimeError(f"bad jwt {token}")
    _db_override(client, db)
    try:
        with caplog.at_level(logging.DEBUG):
            r = client.post("/api/v1/auth/logout", headers=auth_headers)
    finally:
        client.app.dependency_overrides.pop(get_db_handler, None)
    assert r.status_code == 400
    assert r.json()["detail"] == "Logout failed."
    assert "RuntimeError" in caplog.text
    assert token not in caplog.text + r.text
