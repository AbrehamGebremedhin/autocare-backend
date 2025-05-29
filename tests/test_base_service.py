import pytest
from app.services.base_service import BaseService
import asyncio

class DummyWebSocketManager:
    def __init__(self):
        self.messages = []
    async def broadcast(self, message):
        self.messages.append(message)

class DummyService(BaseService):
    async def _perform_action_impl(self, *args, **kwargs):
        return 'performed'
    async def perform_action(self, *args, **kwargs):
        return await self.run_with_notification(self._perform_action_impl, *args, **kwargs)

def test_base_service_perform_action():
    ws = DummyWebSocketManager()
    service = DummyService(websocket_manager=ws)
    result = asyncio.run(service.perform_action())
    assert result == 'performed'
    # Check notifications
    assert ws.messages[0]['event'] == 'task_started'
    assert ws.messages[-1]['event'] == 'task_finished'
