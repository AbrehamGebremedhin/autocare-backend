from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.db import pool
from app.db.crud import BaseCRUD


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ChatSessionCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('Chatsession')

    async def get_for_user(self, session_id: str, user_id: str) -> dict | None:
        try:
            session_id = str(UUID(session_id))  # canonical form, the shape the uuid codec returns
        except ValueError:
            return None  # the uuid column would reject it with a 500
        rows = await self.read({'id': session_id, 'user_id': user_id})
        return rows[0] if rows else None

    # The session lease: busy_until/busy_by are set while a turn or a delete owns the session, across every
    # worker. busy_by is the holder token claim hands out; renew and release only act for that holder, so a
    # holder whose lease expired and was claimed by someone else can neither extend nor free the new lease.
    # Each call is one atomic statement on the database clock; nothing holds a connection in between. A lease
    # nobody releases (a crashed worker) simply expires.

    async def claim(self, session_id: str, user_id: str, seconds: float = 300) -> str | None:
        """Take the session's lease for `seconds` if it belongs to `user_id` and nobody holds it (or the holder's
        lease expired). The new holder token, or None when it is held, missing, or another user's."""
        try:
            session_id = str(UUID(session_id))
        except ValueError:
            return None
        return await pool.fetchval(
            'UPDATE "Chatsession" SET busy_until = now() + make_interval(secs => $3), busy_by = $4'
            ' WHERE id = $1 AND user_id = $2 AND (busy_until IS NULL OR busy_until < now()) RETURNING busy_by',
            session_id, user_id, float(seconds), str(uuid4()),
        )

    async def renew(self, session_id: str, holder: str, seconds: float = 300) -> bool:
        """Push `holder`'s lease `seconds` into the future (long turns). False when `holder` no longer holds it:
        the session was claimed by someone else after the lease expired, released, or deleted."""
        return await pool.fetchval(
            'UPDATE "Chatsession" SET busy_until = now() + make_interval(secs => $3)'
            ' WHERE id = $1 AND busy_by = $2 RETURNING true',
            session_id, holder, float(seconds),
        ) is True

    async def release(self, session_id: str, holder: str) -> None:
        """Free the lease if `holder` still holds it; someone else's lease is left alone."""
        await pool.execute(
            'UPDATE "Chatsession" SET busy_until = NULL, busy_by = NULL WHERE id = $1 AND busy_by = $2',
            session_id, holder,
        )

    async def list_for_user(self, user_id: str) -> list[dict]:
        return await pool.fetch(
            'SELECT id, car_id, title, updated_at, jsonb_array_length(messages) AS message_count'
            ' FROM "Chatsession" WHERE user_id = $1 ORDER BY updated_at DESC',
            user_id,
        )

    async def append_messages(self, session_id: str, messages: list[dict]) -> None:
        # One atomic statement, so concurrent appends both survive (no read in between).
        await pool.execute(
            'UPDATE "Chatsession" SET messages = messages || $2::jsonb, updated_at = now() WHERE id = $1',
            session_id, messages,
        )

    async def update_message(self, session_id: str, message_id: str, patch: dict) -> bool:
        """Merge `patch` into the one message with this id, in a single statement.
        False when the session or the message does not exist (nothing is written)."""
        status = await pool.execute(
            'UPDATE "Chatsession" s'
            ' SET messages = jsonb_set(s.messages, ARRAY[t.idx::text], (s.messages -> t.idx::int) || $3::jsonb),'
            ' updated_at = now()'
            ' FROM (SELECT e.ord - 1 AS idx FROM "Chatsession" c,'
            ' jsonb_array_elements(c.messages) WITH ORDINALITY AS e(elem, ord)'
            " WHERE c.id = $1 AND e.elem ->> 'id' = $2 ORDER BY e.ord LIMIT 1) t"
            ' WHERE s.id = $1',
            session_id, message_id, patch,
        )
        return status == 'UPDATE 1'

    async def save_state(self, session_id: str, state: dict, title: str) -> None:
        await self.update({'id': session_id}, {'state': state, 'title': title, 'updated_at': _now()})
