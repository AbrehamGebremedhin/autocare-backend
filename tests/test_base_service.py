import pytest
from app.services.base_service import BaseService
import asyncio
import time

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
    def dummy_method(self, x):
        return x

def test_base_service_perform_action():
    ws = DummyWebSocketManager()
    service = DummyService(websocket_manager=ws)
    result = asyncio.run(service.perform_action())
    assert result == 'performed'
    # Check notifications
    assert ws.messages[0]['event'] == 'task_started'
    assert ws.messages[-1]['event'] == 'task_finished'

def test_cache_set_get():
    s = DummyService()
    key = s._get_cache_key('dummy_method', 1)
    s._set_cache(key, 42, ttl_seconds=1)
    assert s._get_from_cache(key) == 42
    time.sleep(1.1)
    assert s._get_from_cache(key) is None

def test_is_cache_valid():
    s = DummyService()
    key = s._get_cache_key('dummy_method', 2)
    s._set_cache(key, 99, ttl_seconds=1)
    assert s._is_cache_valid(key)
    time.sleep(1.1)
    assert not s._is_cache_valid(key)
