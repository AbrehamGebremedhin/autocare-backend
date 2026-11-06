"""Accounts: argon2id password hashes, email+password registration and login, Google sign-in.

Emails are stored and compared lower-cased and stripped. argon2 and Google token checks are
CPU-bound or blocking, so they run in a worker thread. Nothing here logs a password, a hash,
an ID token, or the message of an argon2/Google exception (class names only).
"""
import asyncio

import asyncpg
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from google.auth.transport.requests import Request
from google.oauth2 import id_token as google_id_token

from app.CRUD.user_crud import UserCRUD
from app.core.config import get_settings
from app.utils.logger import get_logger_instance

GOOGLE_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}

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


async def user_exists(user_id: str) -> bool:
    return await users.exists(user_id)


def _verify_google_token(token: str) -> dict:
    # Checks the signature (against Google's published keys) and expiry; audience is checked by the caller.
    return google_id_token.verify_oauth2_token(token, Request(), audience=None)


async def _google_identity(token: str) -> tuple[str, str]:
    """(sub, lower-cased email) of a valid Google ID token for one of our client ids, else GoogleAuthError."""
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
    return sub, normalise_email(email)


async def _google_user(sub: str, email: str) -> dict:
    user = await users.get_by_google_sub(sub)
    if user is None:
        # A password account with this email and no Google link: link it and drop its password (see
        # UserCRUD.link_google). Emails are not verified at registration, so the Google-verified owner wins.
        user = await users.link_google(email, sub)
    if user is None:
        user = await users.create_google_user(email, sub)
    return _public(user)


async def login_with_google(token: str) -> dict:
    """The account for a Google ID token, created or linked on first sign-in. Raises GoogleAuthError."""
    sub, email = await _google_identity(token)
    try:
        return await _google_user(sub, email)
    except asyncpg.UniqueViolationError:
        pass  # a concurrent sign-in created or linked the row first: look again
    try:
        return await _google_user(sub, email)
    except asyncpg.UniqueViolationError:
        # The email belongs to an account linked to a different Google user.
        await logger.warning("Google sign-in conflict: email linked to another Google account")
        raise GoogleAuthError() from None
