from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import JSONResponse
from app.api.v1.routes import router as v1_router
from app.utils.websocket import manager as websocket_manager
from app.utils.logger import Logger
from typing import Any
from pydantic import BaseModel

app = FastAPI()
logger = Logger()

class ErrorResponse(BaseModel):
    detail: str
    code: int

# Unified error handler for HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    await logger.error(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=exc.detail, code=exc.status_code).dict(),
    )

# Unified error handler for generic Exception
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    await logger.error(f"Unhandled Exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="Internal server error", code=500).dict(),
    )

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
        except Exception as exc:
            await self.logger.error(f"WebSocket error: {str(exc)}")
            await self.manager.disconnect(websocket)
            await websocket.close(code=1011)

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