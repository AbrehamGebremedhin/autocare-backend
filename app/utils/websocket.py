from app.core.config import get_settings
from fastapi import WebSocket
from typing import List
import asyncio
from app.core.interfaces import IWebSocketManager

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

manager = ConnectionManager()
