import pytest
from unittest.mock import AsyncMock, MagicMock
from app.utils.websocket import ConnectionManager

# Add your tests for websocket here
def test_placeholder():
    assert True

@pytest.mark.asyncio
async def test_connect_and_disconnect():
    manager = ConnectionManager()
    ws = MagicMock()
    ws.accept = AsyncMock()
    await manager.connect(ws)
    assert ws in manager.active_connections
    await manager.disconnect(ws)
    assert ws not in manager.active_connections

@pytest.mark.asyncio
async def test_send_personal_message():
    manager = ConnectionManager()
    ws = MagicMock()
    ws.send_text = AsyncMock()
    await manager.send_personal_message('msg', ws)
    ws.send_text.assert_awaited_with('msg')

@pytest.mark.asyncio
async def test_broadcast():
    manager = ConnectionManager()
    ws1 = MagicMock()
    ws1.send_text = AsyncMock()
    ws2 = MagicMock()
    ws2.send_text = AsyncMock()
    manager.active_connections = [ws1, ws2]
    await manager.broadcast('msg')
    ws1.send_text.assert_awaited_with('msg')
    ws2.send_text.assert_awaited_with('msg')
