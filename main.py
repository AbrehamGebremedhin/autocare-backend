from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from app.api.v1.routes import router as v1_router
from app.utils.websocket import manager as websocket_manager
from app.utils.logger import Logger, get_logger_instance
from app.db.base import SupabaseDBHandler
from app.utils.redis_cache import get_redis_cache, RedisCache
from typing import Any
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI()
logger = get_logger_instance()
db_handler = SupabaseDBHandler()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

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

# Exception handler for rate limiting
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("Rate limit exceeded", status_code=429)

# Dependency for logger
async def get_logger_dep() -> Logger:
    return logger

# Dependency for websocket manager
async def get_websocket_manager_dep() -> Any:
    return websocket_manager

# Dependency for db handler
async def get_db_handler_dep() -> SupabaseDBHandler:
    return db_handler

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
    await logger.info("WebSocket manager is ready.")

@app.on_event("shutdown")
async def shutdown_event():
    await logger.info("WebSocket manager is shutting down.")

app.include_router(v1_router, prefix="/api/v1")

@app.get("/")
@limiter.limit("10/minute")
async def read_root():
    return {"message": "Welcome to AutoCare API"}

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    websocket_manager: Any = Depends(get_websocket_manager_dep),
    logger: Logger = Depends(get_logger_dep),
):
    handler = WebSocketHandler(websocket_manager, logger)
    await handler.handle(websocket)
