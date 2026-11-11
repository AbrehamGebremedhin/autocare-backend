"""Accounts: argon2 hashing, register/login, Google sign-in. Offline: fake user store or fake pool, Google mocked."""
import asyncio
import logging
import threading
import time

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
    assert await auth_service.user_exists(user["id"], time.time()) is True
    assert await auth_service.user_exists(USER, time.time()) is False


async def test_token_validity_boundary_is_exact_not_per_second(fake_users):
    """Chosen boundary: valid iff tokens_valid_after <= iat, both with microseconds. A token minted earlier
    in the same second as the bump is rejected; one minted at or after the bump is accepted."""
    user = fake_users.add("driver@example.com", google_sub="google-sub-1")
    bump = 1_800_000_000.5
    user["tokens_valid_after"] = bump
    assert await auth_service.user_exists(user["id"], bump) is True
    assert await auth_service.user_exists(user["id"], bump + 0.4) is True
    assert await auth_service.user_exists(user["id"], bump - 0.000001) is False
    assert await auth_service.user_exists(user["id"], 1_800_000_000.0) is False  # flooring to the second is not enough


# --- Google sign-in -----------------------------------------------------------

async def test_google_first_sign_in_creates_a_user_without_password(fake_users, google):
    user = await auth_service.login_with_google("google-id-token")
    row = fake_users.rows[user["id"]]
    assert user == {"id": row["id"], "email": "driver@gmail.com"}
    assert row["google_sub"] == "google-sub-1" and row["password_hash"] is None
    assert google.call_args.args[0] == "google-id-token"
    assert google.call_args.kwargs == {"audience": None, "clock_skew_in_seconds": 10}


async def test_google_second_sign_in_with_same_sub_reuses_the_user(fake_users, google):
    first = await auth_service.login_with_google("token-1")
    google.return_value = _claims(email="renamed@gmail.com")  # the sub is what identifies the account
    assert await auth_service.login_with_google("token-2") == first
    assert len(fake_users.rows) == 1


@pytest.mark.parametrize(
    "email, hd",
    [("Driver@Gmail.com", None), ("driver@googlemail.com", None), ("alice@company.com", "company.com")],
)
async def test_google_email_match_takes_over_a_password_account_when_google_is_authoritative(
    fake_users, google, email, hd
):
    """Pre-registration takeover: the Google-verified owner gets the account, the old password stops
    working and every token issued before the link is revoked."""
    google.return_value = _claims(email=email, **({"hd": hd} if hd else {}))
    squatter = await auth_service.register(email, "attacker-pass")
    issued_before = time.time()
    user = await auth_service.login_with_google("google-id-token")
    assert user["id"] == squatter["id"]
    row = fake_users.rows[user["id"]]
    assert row["google_sub"] == "google-sub-1" and row["password_hash"] is None
    assert row["tokens_valid_after"] > issued_before
    assert await auth_service.user_exists(user["id"], issued_before) is False
    assert await auth_service.user_exists(user["id"], time.time()) is True
    assert await auth_service.authenticate(email, "attacker-pass") is None
    assert len(fake_users.rows) == 1


@pytest.mark.parametrize("hd", [None, "", "  "])
async def test_google_does_not_link_an_address_it_is_not_authoritative_for(fake_users, google, hd):
    google.return_value = _claims(email="Alice@Company.com", **({} if hd is None else {"hd": hd}))
    owner = fake_users.add("alice@company.com", password_hash=auth_service.hash_password("alice-pass"))
    before = dict(owner)
    with pytest.raises(GoogleAuthError):
        await auth_service.login_with_google("google-id-token")
    assert fake_users.rows == {owner["id"]: before}
    assert await auth_service.authenticate("alice@company.com", "alice-pass") == {
        "id": owner["id"], "email": "alice@company.com",
    }


async def test_google_creates_a_new_account_for_any_verified_email(fake_users, google):
    google.return_value = _claims(email="Bob@Company.com")
    user = await auth_service.login_with_google("google-id-token")
    row = fake_users.rows[user["id"]]
    assert row["email"] == "bob@company.com" and row["google_sub"] == "google-sub-1"
    assert row["password_hash"] is None


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
    assert "lower(email) = lower($1)" in sql and args == ("driver@example.com",)


async def test_exists_skips_the_query_for_a_non_uuid(fake_pool):
    assert await UserCRUD().exists("user-123", 1.5) is False
    assert fake_pool.calls == []
    fake_pool.value = True
    assert await UserCRUD().exists(USER.upper(), 1_800_000_000.25) is True
    [(method, sql, args)] = fake_pool.calls
    assert method == "fetchval" and "tokens_valid_after <= to_timestamp($2)" in sql
    assert args == (USER, 1_800_000_000.25)


async def test_link_google_sets_sub_and_drops_password_in_one_statement(fake_pool):
    await UserCRUD().link_google("driver@gmail.com", "google-sub-1")
    assert len(fake_pool.calls) == 1
    _, sql, args = fake_pool.calls[0]
    flat = " ".join(sql.split())
    assert "SET google_sub = $2, password_hash = NULL, tokens_valid_after = greatest(tokens_valid_after, now())" in flat
    assert "WHERE lower(email) = lower($1) AND google_sub IS NULL" in flat
    assert args == ("driver@gmail.com", "google-sub-1")


async def test_logout_revokes_a_refresh_that_checked_its_token_before_the_cutoff_committed(fake_users):
    # The race: a refresh reads its clock after logout's first cutoff, then checks the old token before that
    # UPDATE commits (so the check passes). The second cutoff, read after the commit, revokes its new pair.
    user = await auth_service.register("driver@example.com", "s3cret-pass")
    old_iat = time.time() - 1  # the stolen refresh token, minted earlier
    fake_users.rows[user["id"]]["tokens_valid_after"] = old_iat - 1  # the account is older than the token
    real = fake_users.revoke_tokens
    refresh = {}

    async def first_commit_lags(user_id, cutoff):
        if not refresh:
            refresh["iat"] = time.time()  # the refresh's issued_at, after logout read its cutoff
            refresh["passed"] = await fake_users.exists(user_id, old_iat)  # before the UPDATE lands
        await real(user_id, cutoff)

    fake_users.revoke_tokens = first_commit_lags
    await auth_service.log_out_everywhere(user["id"], old_iat)
    assert refresh["passed"] is True
    assert await fake_users.exists(user["id"], refresh["iat"]) is False  # the refreshed pair is revoked
    assert await fake_users.exists(user["id"], time.time()) is True  # the next sign-in works


async def test_logout_cutoff_is_taken_again_after_the_first_update(fake_users, monkeypatch):
    user = await auth_service.register("driver@example.com", "s3cret-pass")
    cutoffs = []
    real = fake_users.revoke_tokens

    async def spy(user_id, cutoff):
        cutoffs.append(cutoff)
        await real(user_id, cutoff)

    monkeypatch.setattr(fake_users, "revoke_tokens", spy)
    await auth_service.log_out_everywhere(user["id"], time.time() - 1)
    assert len(cutoffs) == 2 and cutoffs[0] < cutoffs[1]


async def test_inserts_bind_every_value(fake_pool):
    await UserCRUD().create_password_user("a@example.com", "$argon2id$hash")
    await UserCRUD().create_google_user("b@example.com", "sub-b")
    assert [c[2] for c in fake_pool.calls] == [("a@example.com", "$argon2id$hash"), ("b@example.com", "sub-b")]
    assert all("a@example.com" not in c[1] and "sub-b" not in c[1] for c in fake_pool.calls)


# --- Google certificate cache ---------------------------------------------------

CERTS_URL = "https://www.googleapis.com/oauth2/v1/certs"


class _FakeTransport:
    """Counts calls and answers with a fixed status; remembers the keyword arguments of each call."""

    def __init__(self, status=200):
        self.status, self.calls = status, []

    def __call__(self, url, method="GET", **kwargs):
        self.calls.append(kwargs)
        return type("Resp", (), {"status": self.status, "data": b"{}"})()


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(auth_service.time, "monotonic", lambda: now[0])
    return now


def test_certs_are_fetched_once_within_the_ttl(clock):
    inner = _FakeTransport()
    certs = auth_service._CertRequest(inner)
    first = certs(CERTS_URL, method="GET")
    clock[0] += 3599
    assert certs(CERTS_URL, method="GET") is first and len(inner.calls) == 1


def test_certs_are_fetched_again_after_the_ttl(clock):
    inner = _FakeTransport()
    certs = auth_service._CertRequest(inner)
    certs(CERTS_URL)
    clock[0] += 3601
    certs(CERTS_URL)
    assert len(inner.calls) == 2


def test_a_failed_cert_fetch_is_not_cached(clock):
    inner = _FakeTransport(status=500)
    certs = auth_service._CertRequest(inner)
    certs(CERTS_URL)
    certs(CERTS_URL)
    assert len(inner.calls) == 2 and certs.clear() is False


def test_the_cert_fetch_has_a_10_second_timeout(clock):
    inner = _FakeTransport()
    auth_service._CertRequest(inner)(CERTS_URL, timeout=120)
    assert inner.calls[0]["timeout"] == 10


def test_an_unknown_key_refetches_the_certs_once(mocker, clock, monkeypatch):
    certs = auth_service._CertRequest(_FakeTransport())
    certs(CERTS_URL)  # warm
    monkeypatch.setattr(auth_service, "_certs", certs)
    verify = mocker.patch("google.oauth2.id_token.verify_oauth2_token", side_effect=[ValueError("no key"), _claims()])
    assert auth_service._verify_google_token("t") == _claims()
    assert verify.call_count == 2
    assert certs.clear() is False  # emptied between the two checks and not refilled by the fake verify


def test_a_bad_token_with_a_cold_cache_is_not_retried(mocker, monkeypatch):
    monkeypatch.setattr(auth_service, "_certs", auth_service._CertRequest(_FakeTransport()))
    verify = mocker.patch("google.oauth2.id_token.verify_oauth2_token", side_effect=ValueError("bad"))
    with pytest.raises(ValueError):
        auth_service._verify_google_token("t")
    assert verify.call_count == 1
