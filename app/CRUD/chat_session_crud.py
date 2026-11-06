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
            UUID(session_id)
        except ValueError:
            return None  # the uuid column would reject it with a 500
        rows = await self.read({'id': session_id, 'user_id': user_id})
        return rows[0] if rows else None

    async def list_for_user(self, user_id: str) -> list[dict]:
        return await pool.fetch(
            'SELECT id, car_id, title, updated_at, jsonb_array_length(messages) AS message_count'
            ' FROM "Chatsession" WHERE user_id = $1 ORDER BY updated_at DESC',
            user_id,
        )

    async def append_messages(self, session_id: str, messages: list[dict]) -> None:
        # Read-modify-write; the route's per-session in-flight lock keeps it single-writer.
        rows = await self.read({'id': session_id})
        if not rows:
            return
        await self.update(
            {'id': session_id},
            {'messages': [*(rows[0].get('messages') or []), *messages], 'updated_at': _now()},
        )

    async def update_message(self, session_id: str, message_id: str, patch: dict) -> None:
        # Same read-modify-write (and single-writer lock) as append_messages.
        rows = await self.read({'id': session_id})
        if not rows:
            return
        messages = [{**m, **patch} if m.get('id') == message_id else m for m in rows[0].get('messages') or []]
        await self.update({'id': session_id}, {'messages': messages, 'updated_at': _now()})

    async def save_state(self, session_id: str, state: dict, title: str) -> None:
        await self.update({'id': session_id}, {'state': state, 'title': title, 'updated_at': _now()})
