"""The users table: one row per account, signing in with a password, Google, or both.

Callers pass emails already lower-cased and stripped; lookups compare lower(email) with
lower($1), so they hit the unique index on lower(email) and do not depend on Python's
lower() agreeing with Postgres' for every character. A unique violation (SQLSTATE 23505) propagates
as asyncpg.UniqueViolationError for the caller to map.
"""
from uuid import UUID

from app.db import pool

_COLUMNS = "id, email, password_hash, google_sub"


class UserCRUD:
    async def get_by_email(self, email: str) -> dict | None:
        return await pool.fetchrow(f"SELECT {_COLUMNS} FROM users WHERE lower(email) = lower($1)", email)

    async def get_by_google_sub(self, google_sub: str) -> dict | None:
        return await pool.fetchrow(f"SELECT {_COLUMNS} FROM users WHERE google_sub = $1", google_sub)

    async def exists(self, user_id: str, issued_at: float) -> bool:
        """True when the user exists and accepts a token issued at `issued_at` (epoch seconds, fractional).

        Compared at microsecond precision: a token is valid iff tokens_valid_after <= its iat, so one
        minted earlier in the same second as a bump is rejected and one minted after it is not. iat comes
        from the app's clock and the bump from the database's, so they must share a clock (same host).
        """
        try:
            user_id = str(UUID(user_id))
        except (TypeError, ValueError):
            return False  # the uuid column would reject it with an error, and no such user can exist
        return bool(await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM users WHERE id = $1 AND tokens_valid_after <= to_timestamp($2))",
            user_id,
            float(issued_at),
        ))

    async def create_password_user(self, email: str, password_hash: str) -> dict:
        return await pool.fetchrow(
            "INSERT INTO users (email, password_hash) VALUES ($1, $2) RETURNING id, email", email, password_hash
        )

    async def create_google_user(self, email: str, google_sub: str) -> dict:
        return await pool.fetchrow(
            "INSERT INTO users (email, google_sub) VALUES ($1, $2) RETURNING id, email", email, google_sub
        )

    async def link_google(self, email: str, google_sub: str) -> dict | None:
        """Attach google_sub to the account with this email that has none, drop its password and revoke
        every token issued so far, in one statement. None when no such row exists.

        Emails are never verified at registration, so a password set on this row may belong to whoever
        registered the address first. Only call this when Google is authoritative for the address: the
        owner of that Google account wins and from now on reaches the account through Google only. They
        inherit whatever is already in it (cars, sessions); nothing is wiped.
        """
        return await pool.fetchrow(
            "UPDATE users SET google_sub = $2, password_hash = NULL, tokens_valid_after = now()"
            " WHERE lower(email) = lower($1) AND google_sub IS NULL RETURNING id, email",
            email,
            google_sub,
        )
