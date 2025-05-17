from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.api.v1.routes import router as v1_router
from app.utils.websocket import manager as websocket_manager
from app.utils.logger import Logger
from typing import Any

app = FastAPI()
logger = Logger()

def get_logger() -> Logger:
    return logger

def get_websocket_manager() -> Any:
    return websocket_manager

class WebSocketHandler:
    """
    Handles WebSocket connections and messaging.
    """
    def __init__(self, manager: Any, logger: Logger):
        self.manager = manager
        self.logger = logger

    async def handle(self, websocket: WebSocket) -> None:
        await self.manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await self.manager.send_personal_message(f"You wrote: {data}", websocket)
        except WebSocketDisconnect:
            await self.manager.disconnect(websocket)
            await self.logger.info("WebSocket disconnected.")

@app.on_event("startup")
async def startup_event():
    await get_logger().info("WebSocket manager is ready.")

@app.on_event("shutdown")
async def shutdown_event():
    await get_logger().info("WebSocket manager is shutting down.")

app.include_router(v1_router, prefix="/api/v1")

@app.get("/")
async def read_root():
    return {"message": "Welcome to AutoCare API"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    handler = WebSocketHandler(get_websocket_manager(), get_logger())
    await handler.handle(websocket)