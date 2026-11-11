import logging

import jwt
import pytest
from fastapi.routing import iter_route_contexts

PUBLIC = {"/health", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect"}
SIGNED_DOWNLOAD = "/api/v1/visuals/files/{name:path}"  # public on purpose: the url's HMAC is the credential
TOKEN_KEYS = {"access_token", "refresh_token", "token_type", "expires_in", "user_id"}
CLIENT_ID = "client-1.apps.googleusercontent.com"


def test_every_non_auth_route_requires_token(client):
    for route in iter_route_contexts(client.app.routes):
        path = getattr(route, "path", "")
        if path in PUBLIC or path == SIGNED_DOWNLOAD or path.startswith("/api/v1/auth"):
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


def _logout(client, token):
    return client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})


def _probe(client, token):
    """A protected read that changes nothing: 200 while the token is valid, 401 once it is revoked."""
    return client.get("/api/v1/cars", headers={"Authorization": f"Bearer {token}"})


def _refresh(client, token):
    return client.post("/api/v1/auth/refresh", json={"refresh_token": token})


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


def test_logout_validates_token_and_revokes_the_users_tokens(client, auth_headers, fake_pool, user_id, caplog):
    from app.utils import auth_middleware

    token = auth_headers["Authorization"].split(" ", 1)[1]
    with caplog.at_level(logging.DEBUG):
        r = client.post("/api/v1/auth/logout", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == {"message": "Logged out successfully."}
    auth_middleware.validate_user.assert_awaited_once()
    first, second = fake_pool.calls  # the cutoff, then again with the clock read after it committed
    for method, sql, args in (first, second):
        assert method == "execute" and "SET tokens_valid_after = greatest(tokens_valid_after, to_timestamp($2))" in sql
        assert "WHERE id = $1" in sql and args[0] == user_id
    assert first[2][1] > _claims_of(token)["iat"]
    assert token not in caplog.text
    assert "Logout successful" in caplog.text


def test_logout_signs_out_every_device_and_a_new_login_works(client, fake_users):
    phone = _register(client).json()
    laptop = _login(client).json()
    assert _logout(client, laptop["access_token"]).status_code == 200
    for old in (phone, laptop):
        assert _probe(client, old["access_token"]).status_code == 401
        assert _refresh(client, old["refresh_token"]).status_code == 401
        assert _logout(client, old["access_token"]).status_code == 401
    fresh = _login(client)
    assert fresh.status_code == 200
    assert _probe(client, fresh.json()["access_token"]).status_code == 200
    assert _refresh(client, fresh.json()["refresh_token"]).status_code == 200


def test_logout_leaves_other_users_signed_in(client, fake_users):
    me = _register(client).json()
    other = _register(client, email="other@example.com").json()
    assert _logout(client, me["access_token"]).status_code == 200
    assert _probe(client, other["access_token"]).status_code == 200


def test_a_token_minted_in_the_same_instant_as_the_logout_cutoff_is_valid(client, fake_users, monkeypatch):
    import time

    old = _register(client).json()
    cutoff = max(time.time(), _claims(old)["iat"] + 1e-5)  # not in the future: PyJWT rejects a future iat
    monkeypatch.setattr(time, "time", lambda: cutoff)  # logout and the next login read the same clock value
    assert _logout(client, old["access_token"]).status_code == 200
    assert next(iter(fake_users.rows.values()))["tokens_valid_after"] == cutoff
    fresh = _login(client).json()
    assert _claims(fresh)["iat"] == cutoff
    assert _probe(client, fresh["access_token"]).status_code == 200
    assert _probe(client, old["access_token"]).status_code == 401


def test_logout_revokes_its_own_token_even_if_the_clock_has_not_ticked(client, fake_users, monkeypatch):
    import time

    frozen = time.time()
    monkeypatch.setattr(time, "time", lambda: frozen)  # coarse clock: login and logout get the same reading
    token = _register(client).json()["access_token"]
    assert _logout(client, token).status_code == 200
    assert _probe(client, token).status_code == 401


def test_logout_during_a_database_outage_is_503(client, fake_users, monkeypatch):
    token = _register(client).json()["access_token"]

    async def down(*args):
        from app.db.pool import DatabaseUnavailable

        raise DatabaseUnavailable("ConnectionRefusedError")

    monkeypatch.setattr(fake_users, "revoke_tokens", down)
    r = _logout(client, token)
    assert r.status_code == 503 and r.json()["error_code"] == "SYS_002"
    assert _probe(client, token).status_code == 200  # nothing was revoked


def test_a_bug_in_logout_or_refresh_is_a_500_not_retry_later(fake_users, monkeypatch):
    import main
    from fastapi.testclient import TestClient

    async def bug(*args):
        raise KeyError("issued_at")

    with TestClient(main.app, raise_server_exceptions=False) as c:
        tokens = _register(c).json()
        monkeypatch.setattr(fake_users, "revoke_tokens", bug)
        r = _logout(c, tokens["access_token"])
        assert r.status_code == 500 and r.json()["error_code"] == "SYS_001"
        monkeypatch.setattr(fake_users, "exists", bug)
        r = _refresh(c, tokens["refresh_token"])
        assert r.status_code == 500 and r.json()["error_code"] == "SYS_001"


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




def test_google_takeover_of_a_preregistered_email_locks_out_the_password(client, fake_users, google):
    squatter = _register(client, email="Driver@Gmail.com", password="attacker-pass").json()
    r = _google(client)
    assert r.status_code == 200 and r.json()["user_id"] == squatter["user_id"]
    assert _login(client, email="driver@gmail.com", password="attacker-pass").status_code == 401


def test_google_link_revokes_tokens_issued_before_it(client, fake_users, google):
    import asyncio
    from app.services import auth_service

    squatter = _register(client, email="driver@gmail.com", password="attacker-pass").json()
    assert _probe(client, squatter["access_token"]).status_code == 200  # valid until the link

    victim = _google(client).json()
    assert victim["user_id"] == squatter["user_id"]
    assert _probe(client, squatter["access_token"]).status_code == 401
    assert _probe(client, victim["access_token"]).status_code == 200

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
    assert _probe(client, r.json()["access_token"]).status_code == 401


def test_google_will_not_link_a_non_google_address(client, fake_users, google):
    google.return_value = _google_claims(email="alice@company.com")
    owner = _register(client, email="alice@company.com", password="alice-pass").json()
    r = _google(client)
    assert r.status_code == 401 and r.json()["message"] == "Google sign-in failed"
    assert _login(client, email="alice@company.com", password="alice-pass").json()["user_id"] == owner["user_id"]
    assert _probe(client, owner["access_token"]).status_code == 200


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
    assert _probe(client, body["access_token"]).status_code == 200
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


# --- invitation-only sign-up --------------------------------------------------

@pytest.fixture
def invite_only(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "REGISTRATION_CODE", "open-sesame")


def _register_with(client, invite_code=None, **extra):
    body = {"email": "driver@example.com", "password": "s3cret-pass", **extra}
    if invite_code is not None:
        body["invite_code"] = invite_code
    return client.post("/api/v1/auth/register", json=body)


def test_registration_is_open_without_a_code_setting(client, fake_users):
    assert _register_with(client).status_code == 200
    assert _register_with(client, invite_code="anything", email="two@example.com").status_code == 200


def test_register_with_the_right_code(client, fake_users, invite_only):
    r = _register_with(client, invite_code="open-sesame")
    assert r.status_code == 200 and set(r.json()) == TOKEN_KEYS
    assert len(fake_users.rows) == 1


@pytest.mark.parametrize("code", [None, "", "   ", "wrong", "open-sesam", "open-sesame-and-more", "open-sesamé"])
def test_register_with_a_missing_or_wrong_code_gets_one_generic_403(client, fake_users, invite_only, code):
    r = _register_with(client, invite_code=code)
    assert r.status_code == 403
    body = r.json()
    assert body["error_code"] == "AUTH_004" and body["message"] == "Registration is by invitation."
    assert "open-sesame" not in r.text and fake_users.rows == {}


def test_the_code_is_checked_before_hashing_or_the_database(client, fake_users, invite_only, mocker, fake_pool):
    hashed = mocker.patch("app.services.auth_service.hash_password")
    register = mocker.spy(fake_users, "create_password_user")
    assert _register_with(client, invite_code="wrong").status_code == 403
    assert _register_with(client, password="short").status_code == 403
    hashed.assert_not_called()
    register.assert_not_called()
    assert fake_pool.calls == []


def test_the_code_is_never_logged(client, fake_users, invite_only, caplog):
    with caplog.at_level(logging.DEBUG):
        _register_with(client, invite_code="open-sesame-but-wrong")
        _register_with(client, invite_code="open-sesame")
    assert "open-sesame" not in caplog.text


def test_google_will_not_create_an_account_while_registration_is_by_invitation(client, fake_users, google, invite_only):
    r = _google(client)
    assert r.status_code == 403
    assert r.json()["message"] == "Registration is by invitation."
    assert fake_users.rows == {}


def test_google_still_signs_in_and_links_existing_accounts_while_registration_is_by_invitation(
    client, fake_users, google, invite_only
):
    fake_users.add("driver@gmail.com", google_sub="google-sub-1")
    assert _google(client).status_code == 200
    fake_users.rows.clear()
    fake_users.add("driver@gmail.com", password_hash="x")
    assert _google(client).status_code == 200
    [row] = fake_users.rows.values()
    assert row["google_sub"] == "google-sub-1"


@pytest.mark.parametrize("code", ["open-sesame ", " open-sesame", "open-sesame\n"])
def test_whitespace_around_the_client_code_is_ignored(client, fake_users, invite_only, code):
    assert _register_with(client, invite_code=code).status_code == 200


def test_whitespace_around_the_configured_code_is_ignored(client, fake_users, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "REGISTRATION_CODE", "  open-sesame\n")
    assert _register_with(client, invite_code="open-sesame").status_code == 200
    assert _register_with(client, invite_code="wrong", email="two@example.com").status_code == 403


def test_a_blank_configured_code_leaves_registration_open(client, fake_users, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "REGISTRATION_CODE", "   ")
    assert _register_with(client).status_code == 200


def test_codes_are_compared_as_equal_length_digests(client, fake_users, invite_only, mocker):
    import hmac

    compare = mocker.spy(hmac, "compare_digest")
    _register_with(client, invite_code="x")
    assert [(len(c.args[0]), len(c.args[1])) for c in compare.call_args_list] == [(32, 32)]
