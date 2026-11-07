"""Accounts: argon2id password hashes, email+password registration and login, Google sign-in.

Emails are stored and compared lower-cased and stripped. argon2 and Google token checks are
CPU-bound or blocking, so they run in a worker thread. Nothing here logs a password, a hash,
an ID token, or the message of an argon2/Google exception (class names only).
"""
import asyncio
import time

import asyncpg
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from google.auth.transport.requests import Request
from google.oauth2 import id_token as google_id_token

from app.CRUD.user_crud import UserCRUD
from app.core.config import get_settings
from app.utils.logger import get_logger_instance

GOOGLE_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}
# Google is authoritative for these addresses (and for any Workspace account, which has an hd claim).
# For other addresses a verified email only means the address was verified once, maybe long ago.
GOOGLE_EMAIL_DOMAINS = {"gmail.com", "googlemail.com"}
GOOGLE_CLOCK_SKEW_SECONDS = 10

_hasher = PasswordHasher()  # argon2id with the library's defaults
# Checked when there is no hash to compare with (unknown email, Google-only account), so that
# every password login costs exactly one argon2 verify and timing does not reveal which case it was.
_DUMMY_HASH = _hasher.hash("autocare: no account has this password hash")

users = UserCRUD()
logger = get_logger_instance("auth_service")


class EmailTaken(Exception):
    """An account with this email already exists."""


class GoogleAuthError(Exception):
    """The Google ID token was rejected or cannot be tied to one account. Carries no detail by design."""


class SignupClosed(Exception):
    """A Google sign-in would create an account while registration is closed."""


def normalise_email(email: str) -> str:
    return email.strip().lower()


def _public(user: dict) -> dict:
    return {"id": user["id"], "email": user["email"]}


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """True only for a matching hash. With no hash it still runs one verify (against a dummy) and returns False."""
    try:
        _hasher.verify(_DUMMY_HASH if password_hash is None else password_hash, password)
    except (VerificationError, InvalidHashError):
        return False
    return password_hash is not None


async def register(email: str, password: str) -> dict:
    email = normalise_email(email)
    password_hash = await asyncio.to_thread(hash_password, password)
    try:
        return _public(await users.create_password_user(email, password_hash))
    except asyncpg.UniqueViolationError:
        raise EmailTaken() from None


async def authenticate(email: str, password: str) -> dict | None:
    """The account for this email and password, else None. Unknown email, wrong password and a
    Google-only account (no password_hash) all cost one argon2 verify and look the same."""
    user = await users.get_by_email(normalise_email(email))
    password_hash = user["password_hash"] if user else None
    if not await asyncio.to_thread(verify_password, password, password_hash):
        return None
    return _public(user)


async def user_exists(user_id: str, issued_at: float) -> bool:
    """The user exists and a token of theirs issued at `issued_at` has not been revoked."""
    return await users.exists(user_id, issued_at)


async def log_out_everywhere(user_id: str, token_issued_at: float) -> None:
    """Revoke every token of the user issued so far, on every device; the next sign-in works as usual.

    The cutoff comes from the app's clock, the one that stamps iat, so 'earlier token < cutoff <= next
    sign-in's token' holds whatever the database clock says (Python 3.12 on Windows ticks in ms, Postgres
    in µs). It is at least 10 µs past the calling token's iat (more than the float and µs rounding of
    to_timestamp), so that token dies even when the clock has not ticked since it was minted. A token
    minted in the same instant as the cutoff stays valid (tokens_valid_after <= iat).

    A /auth/refresh running alongside could read its clock after that cutoff yet check its old token before
    the UPDATE commits, and so mint a pair that outlives the logout. A second cutoff, read after the first
    UPDATE has committed, revokes such a pair too; refreshes that check later are refused outright.
    """
    await users.revoke_tokens(user_id, max(time.time(), token_issued_at + 1e-5))
    await users.revoke_tokens(user_id, time.time())


def _verify_google_token(token: str) -> dict:
    # Checks the signature (against Google's published keys) and expiry; audience is checked by the caller.
    return google_id_token.verify_oauth2_token(
        token, Request(), audience=None, clock_skew_in_seconds=GOOGLE_CLOCK_SKEW_SECONDS
    )


async def _google_identity(token: str) -> tuple[str, str, bool]:
    """(sub, lower-cased email, Google is authoritative for the email) of a valid Google ID token for one
    of our client ids, else GoogleAuthError."""
    try:
        claims = await asyncio.to_thread(_verify_google_token, token)
    except Exception as e:
        await logger.warning(f"Google ID token rejected: {type(e).__name__}")
        raise GoogleAuthError() from None
    sub, email = claims.get("sub"), claims.get("email")
    if (
        claims.get("aud") not in get_settings().get_google_client_ids()
        or claims.get("iss") not in GOOGLE_ISSUERS
        or claims.get("email_verified") is not True
        or not isinstance(sub, str)
        or not sub
        or not isinstance(email, str)
        or not email.strip()
    ):
        await logger.warning("Google ID token rejected: claims")
        raise GoogleAuthError()
    email = normalise_email(email)
    hd = claims.get("hd")
    authoritative = email.rsplit("@", 1)[-1] in GOOGLE_EMAIL_DOMAINS or (isinstance(hd, str) and bool(hd.strip()))
    return sub, email, authoritative


async def _google_user(sub: str, email: str, authoritative: bool, allow_signup: bool) -> dict:
    user = await users.get_by_google_sub(sub)
    if user is None and authoritative:
        # A password account with this email and no Google link: link it, drop its password and revoke its
        # tokens (see UserCRUD.link_google). Emails are not verified at registration, so the owner wins.
        user = await users.link_google(email, sub)
    if user is None and not allow_signup:
        raise SignupClosed()
    if user is None:
        # Any verified email may open a new account. If the email already has one that Google cannot
        # vouch for (or that is linked to another Google user), the unique index refuses it: no change.
        user = await users.create_google_user(email, sub)
    return _public(user)


async def login_with_google(token: str, allow_signup: bool = True) -> dict:
    """The account for a Google ID token, created or linked on first sign-in. Raises GoogleAuthError, and
    SignupClosed when it would have to create an account but `allow_signup` is False."""
    identity = await _google_identity(token)
    try:
        return await _google_user(*identity, allow_signup)
    except asyncpg.UniqueViolationError:
        pass  # a concurrent sign-in created or linked the row first: look again
    try:
        return await _google_user(*identity, allow_signup)
    except asyncpg.UniqueViolationError:
        # The email belongs to another account that this Google user may not take over.
        await logger.warning("Google sign-in conflict: email belongs to an account it cannot link")
        raise GoogleAuthError() from None
