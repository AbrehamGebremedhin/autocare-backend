from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from app.api.v1.routes import router as v1_router
from app.utils.websocket import manager as websocket_manager
from app.utils.logger import Logger as AppLogger, get_logger_instance
from app.db.base import SupabaseDBHandler as AppDBHandler
from app.utils.redis_cache import get_redis_cache, RedisCache
from app.utils.startup_checks import check_milvus_connection, check_supabase_connection, check_redis_connection
from app.core.interfaces import ILogger, IDBHandler, IWebSocketManager
from typing import Any
from pydantic import BaseModel
from app.utils.limiter import limiter, rate_limit_middleware
from app.utils.validation_middleware import validation_middleware, input_validator
from app.utils.audit_logging import audit_middleware, audit_logger, AuditEventType
from app.utils.security_middleware import SecurityHeadersMiddleware
from app.utils.exceptions import BaseAPIException, SystemException
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
import asyncio
import os
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
    - Enhanced security with rate limiting, input validation, and audit logging
    - Integration with external services (Milvus, Supabase, Redis)
    """,
    version="1.0.0",
    docs_url="/docs",  # Enable default docs
    redoc_url="/redoc",  # Enable default redoc
    openapi_url="/openapi.json"
)

# Configure security-first CORS
allowed_origins = [
    "http://localhost:3000",  # React dev server
    "http://localhost:8080",  # Local frontend
    "https://autocare.yourdomain.com",  # Production frontend
]

# Get additional origins from environment
if os.getenv("ALLOWED_ORIGINS"):
    additional_origins = os.getenv("ALLOWED_ORIGINS").split(",")
    allowed_origins.extend([origin.strip() for origin in additional_origins])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-CSRF-Token",
    ],
    expose_headers=["X-Total-Count", "X-Rate-Limit-Remaining"],
)

# Add trusted host middleware for additional security
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "autocare.yourdomain.com", "*.yourdomain.com"]
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

app.state.limiter = limiter

# Add security middleware (order matters!)
app.add_middleware(SlowAPIMiddleware)  # Rate limiting (legacy support)

# Add custom middleware
@app.middleware("http")
async def security_middleware_stack(request: Request, call_next):
    """Combined security middleware stack"""
    try:
        # 1. Input validation
        if not await input_validator.validate_request(request):
            raise HTTPException(
                status_code=400,
                detail="Invalid or potentially malicious input detected"
            )
        
        # 2. Rate limiting and audit logging combined
        response = await rate_limit_middleware(request, call_next)
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (like rate limiting)
        raise
    except Exception as e:
        # Log unexpected errors
        await audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            ip_address=request.client.host if request.client else None,
            endpoint=request.url.path,
            method=request.method,
            risk_level="high",
            details={"error": str(e)}
        )
        raise HTTPException(status_code=500, detail="Internal server error")

class ErrorResponse(BaseModel):
    detail: str
    code: int

# Enhanced error handlers with structured error responses
@app.exception_handler(BaseAPIException)
async def api_exception_handler(request: Request, exc: BaseAPIException):
    """Handle custom API exceptions"""
    await audit_logger.log_event(
        event_type=AuditEventType.API_REQUEST,
        ip_address=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        status_code=exc.status_code,
        risk_level="medium" if exc.status_code >= 400 else "low",
        details={
            "error_code": exc.error_code.value,
            "correlation_id": exc.correlation_id
        }
    )
    
    await logger.error(f"API Exception: {exc.error_code.value} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler with audit logging"""
    await audit_logger.log_event(
        event_type=AuditEventType.API_REQUEST,
        ip_address=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        status_code=exc.status_code,
        risk_level="medium" if exc.status_code >= 400 else "low",
        details={"error": exc.detail}
    )
    
    await logger.error(f"HTTPException: {exc.detail}")
    
    # Return structured error response
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    """Rate limit exception handler with enhanced logging"""
    await audit_logger.log_event(
        event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
        ip_address=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        risk_level="medium",
        details={
            "rate_limit": str(exc.detail),
            "retry_after": "60"
        }
    )
    
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
        headers={"Retry-After": "60"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors"""
    correlation_id = getattr(request.state, 'correlation_id', None)
    
    await audit_logger.log_event(
        event_type=AuditEventType.SYSTEM_ERROR,
        ip_address=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        risk_level="high",
        details={
            "error_type": type(exc).__name__,
            "correlation_id": correlation_id
        }
    )
    
    await logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)}")
    
    # Return safe error message without exposing internal details
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "SYS_001",
            "message": "Internal server error",
            "correlation_id": correlation_id,
            "status_code": 500
        }
    )

app.state.limiter = limiter

class ErrorResponse(BaseModel):
    detail: str
    code: int

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
    await logger.info("Application shutting down...")
    
    try:
        # Cleanup resources in proper order
        if hasattr(websocket_manager, "close") and callable(getattr(websocket_manager, "close")):
            await websocket_manager.close()
            await logger.info("WebSocket manager closed")

        # Close Redis connections
        redis_cache = await get_redis_cache()
        if hasattr(redis_cache, "close") and callable(getattr(redis_cache, "close")):
            await redis_cache.close()
            await logger.info("Redis connections closed")

        # Close database connections
        if hasattr(db_handler, "close") and callable(getattr(db_handler, "close")):
            await db_handler.close()
            await logger.info("Database connections closed")

        # Close orchestrator agent
        if hasattr(orchestrator_agent, "close") and callable(getattr(orchestrator_agent, "close")):
            orchestrator_agent.close()
            await logger.info("Orchestrator agent closed")

        # Close logger last
        if hasattr(logger, "close") and callable(getattr(logger, "close")):
            await logger.close()
            
    except Exception as e:
        print(f"Error during shutdown: {str(e)}")  # Use print since logger might be closed

app.include_router(
    v1_router,
    prefix="/api/v1",
    tags=["API v1"],
    responses={404: {"description": "Not found"}},
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
