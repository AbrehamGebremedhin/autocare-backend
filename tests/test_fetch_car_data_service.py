import pytest
import asyncio
from app.services.fetch_car_data_service import FetchCarDataService

class DummyWebSocketManager:
    def __init__(self):
        self.messages = []
    async def broadcast(self, message):
        self.messages.append(message)

@pytest.mark.asyncio
async def test_notify_websocket():
    ws = DummyWebSocketManager()
    service = FetchCarDataService(websocket_manager=ws)
    await service.notify_websocket("hello")
    await asyncio.sleep(0.01)  # Allow event loop to process scheduled task
    assert "hello" in ws.messages

def test_build_url():
    service = FetchCarDataService()
    url = service.build_url("Toyota", "Camry", 2020)
    # Check that the expected substrings are in the dictionary values
    assert any("toyota" in v for v in url.values())
    assert any("camry" in v for v in url.values())
    assert any("2020" in v for v in url.values())

import pytest

def test_build_url_missing_args():
    service = FetchCarDataService()
    with pytest.raises(ValueError):
        service.build_url("", "echo", 2001)
    with pytest.raises(ValueError):
        service.build_url("Toyota", "", 2001)
    with pytest.raises(ValueError):
        service.build_url("Toyota", "echo", None)

@pytest.mark.asyncio
async def test_perform_action_with_notification():
    ws = DummyWebSocketManager()
    class DummyFetchCarDataService(FetchCarDataService):
        async def _perform_action_impl(self, *args, **kwargs):
            return 'done'
        async def perform_action(self, *args, **kwargs):
            return await self.run_with_notification(self._perform_action_impl, *args, **kwargs)
    service = DummyFetchCarDataService(websocket_manager=ws)
    result = await service.perform_action()
    assert result == 'done'
    assert ws.messages[0]['event'] == 'task_started'
    assert ws.messages[-1]['event'] == 'task_finished'
