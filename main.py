from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from app.api.v1.routes import router as v1_router
from app.utils.websocket import manager as websocket_manager
from app.utils.logger import Logger as AppLogger, get_logger_instance
from app.db.base import SupabaseDBHandler as AppDBHandler
from app.utils.redis_cache import get_redis_cache, RedisCache
from app.utils.startup_checks import check_milvus_connection, check_supabase_connection, check_redis_connection
from app.core.interfaces import ILogger, IDBHandler, IWebSocketManager
from typing import Any
from pydantic import BaseModel
from app.utils.limiter import limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
import asyncio
from app.agents.global_agents import orchestrator_agent

# Concrete implementations (adapters)
logger: ILogger = get_logger_instance()
db_handler: IDBHandler = AppDBHandler()
websocket_manager: IWebSocketManager = websocket_manager

app = FastAPI(
    title="AutoCare API",
    description="""
    AutoCare API provides intelligent automotive diagnostics, chat-based assistance, and car data management. 
    Features include:
    - User authentication and management
    - Car CRUD operations
    - Chat sessions with AI assistant for troubleshooting and advice
    - Real-time WebSocket communication
    - Rate limiting and robust error handling
    - Integration with external services (Milvus, Supabase, Redis)
    """,
    version="1.0.0"
)
app.state.limiter = limiter
# Add SlowAPI middleware for rate limiting
app.add_middleware(SlowAPIMiddleware)

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

# Dependency injection (async only, DRY)
async def get_logger_dep() -> ILogger:
    return logger

async def get_websocket_manager_dep() -> IWebSocketManager:
    return websocket_manager

async def get_db_handler_dep() -> IDBHandler:
    return db_handler

# SOLID: Startup checks as a single-responsibility class
class StartupChecker:
    def __init__(self, logger: ILogger):
        self.logger = logger
        self.checks = [
            (check_milvus_connection, "Milvus"),
            (check_supabase_connection, "Supabase"),
            (check_redis_connection, "Redis"),
        ]

    async def run_all(self):
        results = await asyncio.gather(
            *(check() for check, _ in self.checks),
            return_exceptions=True
        )
        for i, (result, (_, name)) in enumerate(zip(results, self.checks)):
            ok, err = (result if not isinstance(result, Exception) else (False, str(result)))
            if not ok:
                await self.logger.error(f"{name} connection failed: {err}")
                raise RuntimeError(f"{name} connection failed: {err}")
            await self.logger.info(f"{name} connection successful.")
        await self.logger.info("WebSocket manager is ready.")

# SOLID: WebSocketHandler depends on abstractions
class WebSocketHandler:
    """
    Handles WebSocket connections and messaging.
    """
    def __init__(self, manager: IWebSocketManager, logger: ILogger):
        self.manager = manager
        self.logger = logger

    async def handle(self, websocket: WebSocket) -> None:
        await self.manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                # Echo back the raw JSON string, not a wrapped message
                await self.manager.send_personal_message(data, websocket)
        except WebSocketDisconnect:
            await self.manager.disconnect(websocket)
            await self.logger.info("WebSocket disconnected.")
        except Exception as exc:
            await self.logger.error(f"WebSocket error: {str(exc)}")
            await self.manager.disconnect(websocket)
            await websocket.close(code=1011)

@app.on_event("startup")
async def startup_event():
    checker = StartupChecker(logger)
    await checker.run_all()

@app.on_event("shutdown")
async def shutdown_event():
    await logger.info("WebSocket manager is shutting down.")
    # Cleanup: close DB, Redis, WebSocket manager, and global agents if possible
    if hasattr(db_handler, "close") and callable(getattr(db_handler, "close")):
        await db_handler.close()
    if hasattr(websocket_manager, "close") and callable(getattr(websocket_manager, "close")):
        await websocket_manager.close()
    if hasattr(logger, "close") and callable(getattr(logger, "close")):
        await logger.close()
    if hasattr(orchestrator_agent, "close") and callable(getattr(orchestrator_agent, "close")):
        orchestrator_agent.close()

app.include_router(
    v1_router,
    prefix="/api/v1",
    tags=["API v1"],
    responses={404: {"description": "Not found"}},
    description="All version 1 API endpoints. See tags for grouping."
)

@app.get(
    "/",
    tags=["Root"],
    summary="API Root Endpoint",
    description="Welcome endpoint for the AutoCare API. Returns a welcome message.",
)
@limiter.limit("10/minute")
async def read_root(request: Request):
    return {"message": "Welcome to AutoCare API"}

@app.websocket(
    "/ws",
    name="WebSocket Endpoint",
)
async def websocket_endpoint(
    websocket: WebSocket,
    websocket_manager: IWebSocketManager = Depends(get_websocket_manager_dep),
    logger: ILogger = Depends(get_logger_dep),
):
    """
    WebSocket endpoint for real-time communication.
    Accepts and echoes messages. Used for chat and live features.
    """
    handler = WebSocketHandler(websocket_manager, logger)
    await handler.handle(websocket)
