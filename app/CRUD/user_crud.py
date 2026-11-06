"""The users table: one row per account, signing in with a password, Google, or both.

Callers pass emails already lower-cased and stripped; lookups use lower(email) so they
hit the unique index on lower(email). A unique violation (SQLSTATE 23505) propagates
as asyncpg.UniqueViolationError for the caller to map.
"""
from uuid import UUID

from app.db import pool

_COLUMNS = "id, email, password_hash, google_sub"


class UserCRUD:
    async def get_by_email(self, email: str) -> dict | None:
        return await pool.fetchrow(f"SELECT {_COLUMNS} FROM users WHERE lower(email) = $1", email)

    async def get_by_google_sub(self, google_sub: str) -> dict | None:
        return await pool.fetchrow(f"SELECT {_COLUMNS} FROM users WHERE google_sub = $1", google_sub)

    async def exists(self, user_id: str) -> bool:
        try:
            user_id = str(UUID(user_id))
        except (TypeError, ValueError):
            return False  # the uuid column would reject it with an error, and no such user can exist
        return bool(await pool.fetchval("SELECT EXISTS (SELECT 1 FROM users WHERE id = $1)", user_id))

    async def create_password_user(self, email: str, password_hash: str) -> dict:
        return await pool.fetchrow(
            "INSERT INTO users (email, password_hash) VALUES ($1, $2) RETURNING id, email", email, password_hash
        )

    async def create_google_user(self, email: str, google_sub: str) -> dict:
        return await pool.fetchrow(
            "INSERT INTO users (email, google_sub) VALUES ($1, $2) RETURNING id, email", email, google_sub
        )

    async def link_google(self, email: str, google_sub: str) -> dict | None:
        """Attach google_sub to the account with this email that has none, and drop its password, in one statement.

        Emails are never verified at registration, so a password set on this row may belong to whoever
        registered the address first. The owner of the Google-verified email wins: from now on the account
        is reachable through Google only. None when no such row exists.
        """
        return await pool.fetchrow(
            "UPDATE users SET google_sub = $2, password_hash = NULL"
            " WHERE lower(email) = $1 AND google_sub IS NULL RETURNING id, email",
            email,
            google_sub,
        )
