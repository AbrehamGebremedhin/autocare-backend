import logging

import jwt
import pytest

PUBLIC = {"/health", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect"}
TOKEN_KEYS = {"access_token", "refresh_token", "token_type", "expires_in", "user_id"}
CLIENT_ID = "client-1.apps.googleusercontent.com"


def test_every_non_auth_route_requires_token(client):
    for route in client.app.routes:
        path = getattr(route, "path", "")
        if path in PUBLIC or path.startswith("/api/v1/auth"):
            continue
        for method in route.methods - {"HEAD", "OPTIONS"}:
            r = client.request(method, path.replace("{", "").replace("}", ""))
            assert r.status_code == 401, (method, path)


def test_debug_and_confirm_auth_endpoints_gone(client):
    for p in [
        "/api/v1/auth/list-users", "/api/v1/auth/manual-confirm", "/api/v1/auth/debug-confirm",
        "/api/v1/auth/confirm", "/api/v1/auth/confirm-page",
    ]:
        assert client.get(p).status_code in (404, 405), p
        assert client.post(p, json={"token": "t"}).status_code in (404, 405), p


def test_jwt_uses_settings_secret(monkeypatch):
    # a token signed with the old hard-coded fallback secret is rejected
    from app.core.config import get_settings
    from app.utils.auth_middleware import jwt_handler, AuthenticationError
    assert jwt_handler.secret_key == get_settings().JWT_SECRET_KEY
    bad = jwt.encode({"sub": "u", "type": "access"}, "your-super-secret-jwt-key-change-in-production", "HS256")
    with pytest.raises(AuthenticationError):
        jwt_handler.verify_token(bad)


def _claims_of(token: str) -> dict:
    from app.utils.auth_middleware import jwt_handler

    return jwt_handler.verify_token(token)


def _claims(token_response: dict) -> dict:
    return _claims_of(token_response["access_token"])


def _register(client, email="driver@example.com", password="s3cret-pass"):
    return client.post("/api/v1/auth/register", json={"email": email, "password": password})


def _login(client, email="driver@example.com", password="s3cret-pass"):
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


# --- register / login ---------------------------------------------------------

def test_register_returns_the_login_token_response(client, fake_users):
    r = _register(client, email="New.Driver@Example.com")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == TOKEN_KEYS and body["token_type"] == "bearer" and body["expires_in"] == 1800
    [row] = fake_users.rows.values()
    assert row["email"] == "new.driver@example.com"
    assert body["user_id"] == row["id"] and _claims(body)["sub"] == row["id"]

    login = _login(client, email="new.driver@example.com")
    assert login.status_code == 200 and set(login.json()) == TOKEN_KEYS
    assert login.json()["user_id"] == row["id"]


def test_register_token_reaches_protected_routes(client, fake_users, mocker):
    mocker.patch("app.api.v1.car_route.user_profile_service.get_user_cars", return_value=[])
    token = _register(client).json()["access_token"]
    assert client.get("/api/v1/cars", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_second_register_of_an_email_is_409(client, fake_users):
    assert _register(client).status_code == 200
    r = _register(client, email="DRIVER@example.com", password="other-pass")
    assert r.status_code == 409
    assert r.json()["error_code"] == "DB_003" and r.json()["message"] == "User already exists"
    assert len(fake_users.rows) == 1


@pytest.mark.parametrize("password", ["short", "x" * 1025])
def test_register_rejects_bad_password_length_without_echoing_it(client, fake_users, password):
    r = _register(client, password=password)
    assert r.status_code == 400
    assert r.json()["message"] == "Password does not meet security requirements"
    assert password not in r.text and fake_users.rows == {}


def test_unknown_email_and_wrong_password_get_the_same_401(client, fake_users, caplog):
    _register(client)
    with caplog.at_level(logging.DEBUG):
        wrong = _login(client, password="wrong-pass-123")
        unknown = _login(client, email="nobody@example.com", password="wrong-pass-123")
    assert wrong.status_code == unknown.status_code == 401
    strip = lambda r: {k: v for k, v in r.json().items() if k not in ("correlation_id", "timestamp")}
    assert strip(wrong) == strip(unknown) == {"error_code": "AUTH_001", "message": "Invalid email or password"}
    assert "wrong-pass-123" not in caplog.text and "s3cret-pass" not in caplog.text
    assert "$argon2" not in caplog.text


def test_login_logs_no_password_hash_or_token(client, fake_users, caplog):
    with caplog.at_level(logging.DEBUG):
        _register(client)
        r = _login(client)
    assert r.status_code == 200
    row = next(iter(fake_users.rows.values()))
    for secret in ("s3cret-pass", row["password_hash"], r.json()["access_token"], r.json()["refresh_token"]):
        assert secret not in caplog.text
    assert "Login successful" in caplog.text


def test_database_errors_are_generic_500_and_log_only_the_class(client, fake_pool, caplog):
    secret = "pg: password=hunter2 host=10.0.0.5"
    fake_pool.error = RuntimeError(secret)
    with caplog.at_level(logging.DEBUG):
        r1 = _register(client)
        r2 = _login(client)
    assert r1.status_code == r2.status_code == 500
    assert "RuntimeError" in caplog.text
    assert secret not in caplog.text + r1.text + r2.text


# --- logout -------------------------------------------------------------------

def test_logout_requires_bearer_header_not_query(client):
    r = client.post("/api/v1/auth/logout", params={"token": "abc"})
    assert r.status_code == 401


def test_logout_validates_token_and_tells_the_client_to_discard_it(client, auth_headers, caplog):
    from app.utils import auth_middleware

    token = auth_headers["Authorization"].split(" ", 1)[1]
    with caplog.at_level(logging.DEBUG):
        r = client.post("/api/v1/auth/logout", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == {"message": "Logged out successfully."}
    auth_middleware.validate_user.assert_awaited_once()
    assert token not in caplog.text
    assert "Logout successful" in caplog.text


def test_logout_with_invalid_token_is_401(client):
    r = client.post("/api/v1/auth/logout", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401


def test_token_of_a_deleted_user_is_401(client, fake_users):
    token = _register(client).json()["access_token"]
    fake_users.rows.clear()
    r = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


# --- Google sign-in -----------------------------------------------------------

@pytest.fixture
def google_on(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "GOOGLE_CLIENT_IDS", CLIENT_ID)


def _google_claims(**overrides) -> dict:
    return {
        "iss": "accounts.google.com", "aud": CLIENT_ID, "sub": "google-sub-1",
        "email": "driver@gmail.com", "email_verified": True, **overrides,
    }


@pytest.fixture
def google(mocker, google_on):
    return mocker.patch("google.oauth2.id_token.verify_oauth2_token", return_value=_google_claims())


def _google(client, token="eyJ-google-id-token"):
    return client.post("/api/v1/auth/google", json={"id_token": token})


def test_google_is_404_when_no_client_ids(client, fake_users, mocker, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "GOOGLE_CLIENT_IDS", "")
    verify = mocker.patch("google.oauth2.id_token.verify_oauth2_token")
    assert _google(client).status_code == 404
    verify.assert_not_called()


def test_google_sign_in_returns_the_login_token_response(client, fake_users, google, caplog):
    with caplog.at_level(logging.DEBUG):
        r = _google(client)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == TOKEN_KEYS and body["token_type"] == "bearer"
    [row] = fake_users.rows.values()
    assert row["password_hash"] is None and row["google_sub"] == "google-sub-1"
    assert body["user_id"] == row["id"] and _claims(body)["sub"] == row["id"]
    assert "eyJ-google-id-token" not in r.text + caplog.text

    again = _google(client, token="another-token")
    assert again.status_code == 200 and again.json()["user_id"] == row["id"]
    assert len(fake_users.rows) == 1


def _logout(client, token):
    return client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})


def test_google_takeover_of_a_preregistered_email_locks_out_the_password(client, fake_users, google):
    squatter = _register(client, email="Driver@Gmail.com", password="attacker-pass").json()
    r = _google(client)
    assert r.status_code == 200 and r.json()["user_id"] == squatter["user_id"]
    assert _login(client, email="driver@gmail.com", password="attacker-pass").status_code == 401


def test_google_link_revokes_tokens_issued_before_it(client, fake_users, google):
    import asyncio
    from app.services import auth_service

    squatter = _register(client, email="driver@gmail.com", password="attacker-pass").json()
    assert _logout(client, squatter["access_token"]).status_code == 200  # valid until the link

    victim = _google(client).json()
    assert victim["user_id"] == squatter["user_id"]
    assert _logout(client, squatter["access_token"]).status_code == 401
    assert _logout(client, victim["access_token"]).status_code == 200

    # Refresh tokens carry iat too, so the refresh route rejects the squatter's and accepts the victim's.
    old_refresh, new_refresh = _claims_of(squatter["refresh_token"]), _claims_of(victim["refresh_token"])
    assert old_refresh["type"] == new_refresh["type"] == "refresh"
    assert asyncio.run(auth_service.user_exists(old_refresh["sub"], old_refresh["iat"])) is False
    assert asyncio.run(auth_service.user_exists(new_refresh["sub"], new_refresh["iat"])) is True


def test_password_login_racing_a_google_link_gets_a_revoked_token(client, fake_users, monkeypatch):
    # The link commits after login read the old hash but before argon2 finished: the login still succeeds,
    # but its tokens must predate the link (iat is stamped before the read), so they are already revoked.
    _register(client, email="driver@gmail.com", password="attacker-pass")
    read = fake_users.get_by_email

    async def read_then_link(email):
        row = await read(email)
        await fake_users.link_google(email, "google-sub-1")
        return row

    monkeypatch.setattr(fake_users, "get_by_email", read_then_link)
    r = _login(client, email="driver@gmail.com", password="attacker-pass")
    assert r.status_code == 200
    assert _logout(client, r.json()["access_token"]).status_code == 401


def test_google_will_not_link_a_non_google_address(client, fake_users, google):
    google.return_value = _google_claims(email="alice@company.com")
    owner = _register(client, email="alice@company.com", password="alice-pass").json()
    r = _google(client)
    assert r.status_code == 401 and r.json()["message"] == "Google sign-in failed"
    assert _login(client, email="alice@company.com", password="alice-pass").json()["user_id"] == owner["user_id"]
    assert _logout(client, owner["access_token"]).status_code == 200


@pytest.mark.parametrize(
    "claims",
    [
        _google_claims(aud="someone-elses-client"),
        _google_claims(iss="https://evil.example.com"),
        _google_claims(email_verified=False),
        {k: v for k, v in _google_claims().items() if k != "sub"},
    ],
)
def test_google_bad_claims_are_generic_401(client, fake_users, google, claims):
    google.return_value = claims
    r = _google(client)
    assert r.status_code == 401
    assert r.json()["message"] == "Google sign-in failed"
    assert "eyJ-google-id-token" not in r.text and fake_users.rows == {}


def test_google_expired_token_is_401_without_the_error_text(client, fake_users, google, caplog):
    google.side_effect = ValueError("Token expired, 1700000000 < 1800000000")
    with caplog.at_level(logging.DEBUG):
        r = _google(client)
    assert r.status_code == 401 and r.json()["message"] == "Google sign-in failed"
    assert "expired" not in (r.text + caplog.text).lower()
    assert "eyJ-google-id-token" not in r.text + caplog.text


def test_google_library_issuer_error_is_generic_401(client, fake_users, google):
    from google.auth import exceptions

    google.side_effect = exceptions.GoogleAuthError("Wrong issuer. 'iss' should be one of the following: ...")
    r = _google(client)
    assert r.status_code == 401 and r.json()["message"] == "Google sign-in failed"
    assert "issuer" not in r.text.lower() and fake_users.rows == {}


def test_google_rejects_an_oversized_token_without_verifying(client, fake_users, google):
    r = _google(client, token="x" * 5000)
    assert r.status_code == 401
    google.assert_not_called()


def test_google_cannot_reach_an_account_linked_to_another_google_user(client, fake_users, google):
    fake_users.add("driver@gmail.com", google_sub="someone-else")
    r = _google(client)
    assert r.status_code == 401
    assert next(iter(fake_users.rows.values()))["google_sub"] == "someone-else"


# --- refresh ------------------------------------------------------------------

REFRESH_401 = {"error_code": "AUTH_001", "message": "Invalid refresh token"}


def _refresh(client, token):
    return client.post("/api/v1/auth/refresh", json={"refresh_token": token})


def _strip(r) -> dict:
    return {k: v for k, v in r.json().items() if k not in ("correlation_id", "timestamp")}


def test_refresh_returns_a_new_working_token_pair(client, fake_users):
    first = _register(client).json()
    r = _refresh(client, first["refresh_token"])
    assert r.status_code == 200
    body = r.json()
    assert set(body) == TOKEN_KEYS and body["user_id"] == first["user_id"] and body["expires_in"] == 1800
    assert _claims_of(body["access_token"])["type"] == "access"
    assert _claims_of(body["refresh_token"])["type"] == "refresh"
    assert _logout(client, body["access_token"]).status_code == 200
    assert _refresh(client, body["refresh_token"]).status_code == 200


def test_refresh_rejects_an_access_token(client, fake_users):
    access = _register(client).json()["access_token"]
    r = _refresh(client, access)
    assert r.status_code == 401 and _strip(r) == REFRESH_401


def test_refresh_token_issued_before_a_google_link_is_401(client, fake_users, google):
    squatter = _register(client, email="driver@gmail.com", password="attacker-pass").json()
    assert _google(client).status_code == 200  # links the account and revokes earlier tokens
    r = _refresh(client, squatter["refresh_token"])
    assert r.status_code == 401 and _strip(r) == REFRESH_401


def test_refresh_rejects_bad_expired_and_foreign_tokens_alike(client, fake_users, caplog):
    import time

    from app.utils.auth_middleware import jwt_handler

    user_id = _register(client).json()["user_id"]
    claims = {"sub": user_id, "type": "refresh", "iat": time.time()}
    expired = jwt.encode({**claims, "exp": time.time() - 60}, jwt_handler.secret_key, "HS256")
    foreign = jwt.encode({**claims, "exp": time.time() + 600}, "another-secret-0123456789-abcdefghijk", "HS256")
    no_iat = jwt.encode({"sub": user_id, "type": "refresh", "exp": time.time() + 600}, jwt_handler.secret_key, "HS256")
    with caplog.at_level(logging.DEBUG):
        responses = [_refresh(client, t) for t in ("not-a-jwt", expired, foreign, no_iat)]
    for token, r in zip(("not-a-jwt", expired, foreign, no_iat), responses):
        assert r.status_code == 401 and _strip(r) == REFRESH_401
        assert token not in caplog.text


def test_refresh_of_a_deleted_user_is_401(client, fake_users):
    refresh = _register(client).json()["refresh_token"]
    fake_users.rows.clear()
    assert _refresh(client, refresh).status_code == 401


def test_refresh_during_a_database_outage_is_503(client, fake_users, fake_pool, monkeypatch):
    from app.CRUD.user_crud import UserCRUD
    from app.services import auth_service

    refresh = _register(client).json()["refresh_token"]
    monkeypatch.setattr(auth_service, "users", UserCRUD())  # the real CRUD, on the failing fake pool
    fake_pool.error = ConnectionError("db down")
    r = _refresh(client, refresh)
    assert r.status_code == 503 and r.json()["error_code"] == "SYS_002"


def test_refresh_never_logs_the_token(client, fake_users, caplog):
    refresh = _register(client).json()["refresh_token"]
    with caplog.at_level(logging.DEBUG):
        r = _refresh(client, refresh)
    assert r.status_code == 200
    for secret in (refresh, r.json()["access_token"], r.json()["refresh_token"]):
        assert secret not in caplog.text


# --- rate limits --------------------------------------------------------------

@pytest.mark.parametrize(
    "path, body",
    [
        ("/api/v1/auth/login", {"email": "nobody@example.com", "password": "s3cret-pass"}),
        ("/api/v1/auth/register", {"email": "a@example.com", "password": "short"}),
        ("/api/v1/auth/google", {"id_token": "t"}),
        ("/api/v1/auth/refresh", {"refresh_token": "t"}),
    ],
)
def test_sign_in_routes_are_rate_limited(client, fake_users, path, body):
    statuses = [client.post(path, json=body).status_code for _ in range(11)]
    assert 429 not in statuses[:10]
    assert statuses[10] == 429
