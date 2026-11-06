"""Accounts: argon2 hashing, register/login, Google sign-in. Offline: fake user store or fake pool, Google mocked."""
import asyncio
import logging
import threading

import asyncpg
import pytest
from argon2 import PasswordHasher

from app.CRUD.user_crud import UserCRUD
from app.core.config import get_settings
from app.services import auth_service
from app.services.auth_service import EmailTaken, GoogleAuthError

CLIENT_ID = "client-1.apps.googleusercontent.com"
USER = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def google_ids(monkeypatch):
    monkeypatch.setattr(get_settings(), "GOOGLE_CLIENT_IDS", f"{CLIENT_ID}, other-client")


def _claims(**overrides) -> dict:
    return {
        "iss": "https://accounts.google.com", "aud": CLIENT_ID, "sub": "google-sub-1",
        "email": "Driver@Gmail.com", "email_verified": True, **overrides,
    }


@pytest.fixture
def google(mocker, google_ids):
    """verify_oauth2_token is the only Google call; it returns these claims unless a test changes them."""
    return mocker.patch("google.oauth2.id_token.verify_oauth2_token", return_value=_claims())


@pytest.fixture
def verify_spy(mocker):
    return mocker.spy(PasswordHasher, "verify")


# --- hashing ------------------------------------------------------------------

def test_hash_and_verify_round_trip():
    h = auth_service.hash_password("correct horse")
    assert h.startswith("$argon2id$") and "correct horse" not in h
    assert auth_service.verify_password("correct horse", h) is True
    assert auth_service.verify_password("wrong horse", h) is False


def test_verify_without_hash_still_runs_argon2_once_and_fails(verify_spy):
    assert auth_service.verify_password("anything", None) is False
    assert verify_spy.call_count == 1


@pytest.mark.parametrize("bad_hash", ["", "not-a-hash", "$argon2id$v=19$broken"])
def test_verify_against_a_malformed_hash_is_false(bad_hash):
    assert auth_service.verify_password("anything", bad_hash) is False


# --- register -----------------------------------------------------------------

async def test_register_normalises_email_and_stores_only_a_hash(fake_users):
    user = await auth_service.register("  New.Driver@Example.COM ", "s3cret-pass")
    row = fake_users.rows[user["id"]]
    assert user == {"id": row["id"], "email": "new.driver@example.com"}
    assert row["email"] == "new.driver@example.com"
    assert row["password_hash"].startswith("$argon2id$") and "s3cret-pass" not in row["password_hash"]
    assert row["google_sub"] is None


async def test_register_duplicate_email_in_any_case_is_email_taken(fake_users):
    await auth_service.register("driver@example.com", "s3cret-pass")
    with pytest.raises(EmailTaken):
        await auth_service.register("DRIVER@example.com", "other-pass")
    assert len(fake_users.rows) == 1


async def test_register_maps_a_raised_unique_violation_to_email_taken(fake_pool):
    fake_pool.error = asyncpg.UniqueViolationError("duplicate key value violates unique constraint")
    with pytest.raises(EmailTaken):
        await auth_service.register("driver@example.com", "s3cret-pass")


async def test_concurrent_double_register_one_wins_one_is_email_taken(fake_users):
    results = await asyncio.gather(
        auth_service.register("race@example.com", "s3cret-pass"),
        auth_service.register("Race@example.com", "s3cret-pass"),
        return_exceptions=True,
    )
    assert sum(isinstance(r, dict) for r in results) == 1
    assert sum(isinstance(r, EmailTaken) for r in results) == 1
    assert len(fake_users.rows) == 1


async def test_argon2_runs_off_the_event_loop(fake_users, monkeypatch):
    threads = []
    hash_password, verify_password = auth_service.hash_password, auth_service.verify_password

    def recording(fn):
        def wrapper(*args):
            threads.append(threading.current_thread())
            return fn(*args)
        return wrapper

    monkeypatch.setattr(auth_service, "hash_password", recording(hash_password))
    monkeypatch.setattr(auth_service, "verify_password", recording(verify_password))
    await auth_service.register("driver@example.com", "s3cret-pass")
    await auth_service.authenticate("driver@example.com", "s3cret-pass")
    await auth_service.authenticate("nobody@example.com", "s3cret-pass")
    assert len(threads) == 3
    assert all(t is not threading.main_thread() for t in threads)


# --- authenticate -------------------------------------------------------------

async def test_authenticate_with_the_right_password(fake_users):
    user = await auth_service.register("driver@example.com", "s3cret-pass")
    assert await auth_service.authenticate(" Driver@Example.com", "s3cret-pass") == user


@pytest.mark.parametrize("email, password", [("driver@example.com", "wrong-pass"), ("nobody@example.com", "s3cret-pass")])
async def test_wrong_password_and_unknown_email_both_fail_after_one_verify(fake_users, verify_spy, email, password):
    fake_users.add("driver@example.com", password_hash=auth_service.hash_password("s3cret-pass"))
    assert await auth_service.authenticate(email, password) is None
    assert verify_spy.call_count == 1


async def test_google_only_account_never_logs_in_by_password(fake_users, verify_spy):
    fake_users.add("driver@example.com", google_sub="google-sub-1")
    assert await auth_service.authenticate("driver@example.com", "") is None
    assert verify_spy.call_count == 1


async def test_user_exists(fake_users):
    user = fake_users.add("driver@example.com", google_sub="google-sub-1")
    assert await auth_service.user_exists(user["id"]) is True
    assert await auth_service.user_exists(USER) is False


# --- Google sign-in -----------------------------------------------------------

async def test_google_first_sign_in_creates_a_user_without_password(fake_users, google):
    user = await auth_service.login_with_google("google-id-token")
    row = fake_users.rows[user["id"]]
    assert user == {"id": row["id"], "email": "driver@gmail.com"}
    assert row["google_sub"] == "google-sub-1" and row["password_hash"] is None
    assert google.call_args.args[0] == "google-id-token"
    assert google.call_args.kwargs == {"audience": None}


async def test_google_second_sign_in_with_same_sub_reuses_the_user(fake_users, google):
    first = await auth_service.login_with_google("token-1")
    google.return_value = _claims(email="renamed@gmail.com")  # the sub is what identifies the account
    assert await auth_service.login_with_google("token-2") == first
    assert len(fake_users.rows) == 1


async def test_google_email_match_takes_over_a_password_account(fake_users, google):
    """Pre-registration takeover: the Google-verified owner gets the account and the old password stops working."""
    squatter = await auth_service.register("driver@gmail.com", "attacker-pass")
    user = await auth_service.login_with_google("google-id-token")
    assert user["id"] == squatter["id"]
    row = fake_users.rows[user["id"]]
    assert row["google_sub"] == "google-sub-1" and row["password_hash"] is None
    assert await auth_service.authenticate("driver@gmail.com", "attacker-pass") is None
    assert len(fake_users.rows) == 1


async def test_google_cannot_sign_into_an_account_linked_to_another_google_user(fake_users, google):
    other = fake_users.add("driver@gmail.com", google_sub="someone-else")
    with pytest.raises(GoogleAuthError):
        await auth_service.login_with_google("google-id-token")
    assert fake_users.rows == {other["id"]: other}
    assert other["google_sub"] == "someone-else"


async def test_google_insert_race_retries_the_lookup_once(fake_users, google, monkeypatch):
    real_create = fake_users.create_google_user
    calls = []

    async def lose_the_race(email, google_sub):
        calls.append(email)
        await real_create(email, google_sub)  # a concurrent sign-in inserts the same user first
        raise asyncpg.UniqueViolationError("duplicate key value violates unique constraint")

    monkeypatch.setattr(fake_users, "create_google_user", lose_the_race)
    user = await auth_service.login_with_google("google-id-token")
    assert len(calls) == 1 and len(fake_users.rows) == 1
    assert fake_users.rows[user["id"]]["google_sub"] == "google-sub-1"


@pytest.mark.parametrize(
    "claims",
    [
        _claims(aud="someone-elses-client"),
        _claims(aud=[CLIENT_ID]),
        _claims(iss="https://evil.example.com"),
        _claims(email_verified=False),
        _claims(email_verified="true"),
        {k: v for k, v in _claims().items() if k != "email_verified"},
        _claims(sub=""),
        {k: v for k, v in _claims().items() if k != "sub"},
        _claims(email=""),
        _claims(email=None),
    ],
)
async def test_google_bad_claims_are_rejected_and_create_nothing(fake_users, google, claims):
    google.return_value = claims
    with pytest.raises(GoogleAuthError):
        await auth_service.login_with_google("google-id-token")
    assert fake_users.rows == {}


async def test_google_verification_failure_logs_the_class_only(fake_users, google, caplog):
    google.side_effect = ValueError("Token expired, 1700000000 < now; eyJ-secret-body")
    with caplog.at_level(logging.DEBUG), pytest.raises(GoogleAuthError) as exc_info:
        await auth_service.login_with_google("eyJ-google-id-token")
    assert "ValueError" in caplog.text
    for secret in ("eyJ-secret-body", "eyJ-google-id-token", "expired"):
        assert secret not in caplog.text and secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None and exc_info.value.__suppress_context__
    assert fake_users.rows == {}


async def test_google_is_off_without_client_ids(fake_users, google, monkeypatch):
    monkeypatch.setattr(get_settings(), "GOOGLE_CLIENT_IDS", " , ")
    assert get_settings().get_google_client_ids() == []
    with pytest.raises(GoogleAuthError):
        await auth_service.login_with_google("google-id-token")


async def test_google_verification_runs_off_the_event_loop(fake_users, google):
    threads = []
    google.side_effect = lambda *a, **k: threads.append(threading.current_thread()) or _claims()
    await auth_service.login_with_google("google-id-token")
    assert threads and threads[0] is not threading.main_thread()


# --- UserCRUD SQL (fake pool) -------------------------------------------------

async def test_email_lookup_uses_the_lower_email_index(fake_pool):
    await UserCRUD().get_by_email("driver@example.com")
    method, sql, args = fake_pool.calls[0]
    assert "lower(email) = $1" in sql and args == ("driver@example.com",)


async def test_exists_skips_the_query_for_a_non_uuid(fake_pool):
    assert await UserCRUD().exists("user-123") is False
    assert fake_pool.calls == []
    fake_pool.value = True
    assert await UserCRUD().exists(USER.upper()) is True
    assert fake_pool.calls[0][2] == (USER,)


async def test_link_google_sets_sub_and_drops_password_in_one_statement(fake_pool):
    await UserCRUD().link_google("driver@gmail.com", "google-sub-1")
    assert len(fake_pool.calls) == 1
    _, sql, args = fake_pool.calls[0]
    flat = " ".join(sql.split())
    assert "SET google_sub = $2, password_hash = NULL" in flat
    assert "WHERE lower(email) = $1 AND google_sub IS NULL" in flat
    assert args == ("driver@gmail.com", "google-sub-1")


async def test_inserts_bind_every_value(fake_pool):
    await UserCRUD().create_password_user("a@example.com", "$argon2id$hash")
    await UserCRUD().create_google_user("b@example.com", "sub-b")
    assert [c[2] for c in fake_pool.calls] == [("a@example.com", "$argon2id$hash"), ("b@example.com", "sub-b")]
    assert all("a@example.com" not in c[1] and "sub-b" not in c[1] for c in fake_pool.calls)
