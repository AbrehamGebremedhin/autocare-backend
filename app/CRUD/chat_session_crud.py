import asyncio
from datetime import datetime, timezone
from uuid import UUID

from app.db.crud import BaseCRUD


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatSessionCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('Chatsession')

    async def unique_logic(self, *args, **kwargs):
        # Implement any Chat_Session-specific logic here
        pass

    async def get_for_user(self, session_id: str, user_id: str) -> dict | None:
        try:
            UUID(session_id)
        except ValueError:
            return None  # the uuid column would reject it with a 500
        rows = await self.read({'id': session_id, 'user_id': user_id})
        return rows[0] if rows else None

    async def list_for_user(self, user_id: str) -> list[dict]:
        async with self.db_handler.get_connection() as db:
            query = (
                db.table(self.table_name)
                .select('id,car_id,title,updated_at,messages')
                .eq('user_id', user_id)
                .order('updated_at', desc=True)
            )
            response = await asyncio.to_thread(query.execute)
        return [
            {**{k: r[k] for k in ('id', 'car_id', 'title', 'updated_at')}, 'message_count': len(r.get('messages') or [])}
            for r in response.data or []
        ]

    async def append_messages(self, session_id: str, messages: list[dict]) -> None:
        # Read-modify-write; the route's per-session in-flight lock keeps it single-writer.
        rows = await self.read({'id': session_id})
        if not rows:
            return
        await self.update(
            {'id': session_id},
            {'messages': [*(rows[0].get('messages') or []), *messages], 'updated_at': _now()},
        )

    async def save_state(self, session_id: str, state: dict, title: str) -> None:
        await self.update({'id': session_id}, {'state': state, 'title': title, 'updated_at': _now()})
