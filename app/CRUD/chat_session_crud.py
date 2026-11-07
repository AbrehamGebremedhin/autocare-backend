from datetime import datetime, timezone
from uuid import UUID

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

    # The session lease: busy_until is set while a turn or a delete owns the session, across every worker.
    # Each call is one atomic statement on the database clock; nothing holds a connection in between. A lease
    # nobody releases (a crashed worker) simply expires.

    async def claim(self, session_id: str, user_id: str, seconds: float = 300) -> bool:
        """Take the session's lease for `seconds` if it belongs to `user_id` and nobody holds it (or the holder's
        lease expired). False when it is held, missing, or another user's."""
        try:
            session_id = str(UUID(session_id))
        except ValueError:
            return False
        return await pool.fetchval(
            'UPDATE "Chatsession" SET busy_until = now() + make_interval(secs => $3)'
            ' WHERE id = $1 AND user_id = $2 AND (busy_until IS NULL OR busy_until < now()) RETURNING true',
            session_id, user_id, float(seconds),
        ) is True

    async def renew(self, session_id: str, seconds: float = 300) -> bool:
        """Push a held lease `seconds` into the future (long turns). False when no lease is held."""
        return await pool.fetchval(
            'UPDATE "Chatsession" SET busy_until = now() + make_interval(secs => $2)'
            ' WHERE id = $1 AND busy_until IS NOT NULL RETURNING true',
            session_id, float(seconds),
        ) is True

    async def release(self, session_id: str) -> None:
        await pool.execute('UPDATE "Chatsession" SET busy_until = NULL WHERE id = $1', session_id)

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
