from app.core.config import get_settings
from fastapi import WebSocket
from typing import List
import asyncio
from app.core.interfaces import IWebSocketManager
from .message_formatter import MessageFormatter
from .message_types import MessageType, MessageSource
import json

# Load settings from config
settings = get_settings()
WEBSOCKET_URL = settings.WEBSOCKET_URL

class ConnectionManager(IWebSocketManager):
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        async with self._lock:
            connections = list(self.active_connections)
        for connection in connections:
            await connection.send_text(message)

    async def send_info(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.INFO, source=source, content=content, session_id=session_id, details=details)
        await websocket.send_text(json.dumps(msg))

    async def send_warning(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.WARNING, source=source, content=content, session_id=session_id, details=details)
        await websocket.send_text(json.dumps(msg))

    async def send_error(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.ERROR, source=source, content=content, session_id=session_id, details=details)
        await websocket.send_text(json.dumps(msg))

    async def send_progress(self, websocket: WebSocket, content: str, source: MessageSource, progress: float, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.PROGRESS, source=source, content=content, progress=progress, session_id=session_id, details=details)
        await websocket.send_text(json.dumps(msg))

    async def send_stage(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.STAGE, source=source, content=content, session_id=session_id, details=details)
        await websocket.send_text(json.dumps(msg))

    async def send_result(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.RESULT, source=source, content=content, session_id=session_id, details=details)
        await websocket.send_text(json.dumps(msg))

    async def send_debug(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.DEBUG, source=source, content=content, session_id=session_id, details=details)
        await websocket.send_text(json.dumps(msg))

    async def broadcast_json(self, msg: dict):
        import json
        await self.broadcast(json.dumps(msg))

    async def broadcast_standard(self, *, type: MessageType, source: MessageSource, content: str, session_id: str = None, progress: float = None, details: dict = None):
        msg = MessageFormatter.format(type=type, source=source, content=content, session_id=session_id, progress=progress, details=details)
        await self.broadcast(json.dumps(msg))

manager = ConnectionManager()
