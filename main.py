"""
AutoCare Backend - Concurrent Version
Optimized for multiple concurrent users with enhanced session management,
connection pooling, and distributed caching.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.api.v1.routes import router as v1_router
from app.utils.concurrent_websocket import ConcurrentWebSocketManager
from app.utils.logger import Logger as AppLogger, get_logger_instance
from app.db.enhanced_connection_pool import ConcurrentDBConnectionPool
from app.utils.redis_cache import get_redis_cache
from app.utils.session_manager import ConcurrentSessionManager
from app.services.concurrent_chat_service import ConcurrentChatService
from app.agents.agent_pool import ConcurrentAgentPool
from app.core.config import get_settings
from app.utils.startup_checks import check_milvus_connection, check_supabase_connection, check_redis_connection
import json
from datetime import datetime
from app.core.interfaces import ILogger, IDBHandler, IWebSocketManager
from app.utils.message_types import MessageSource
from typing import Any, Dict
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
import logging

# Initialize settings
settings = get_settings()

# Global component instances
logger: ILogger = get_logger_instance()
db_pool: ConcurrentDBConnectionPool = None
websocket_manager: ConcurrentWebSocketManager = None
session_manager: ConcurrentSessionManager = None
chat_service: ConcurrentChatService = None
agent_pool: ConcurrentAgentPool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan management for concurrent system components"""
    global db_pool, websocket_manager, session_manager, chat_service, agent_pool
    
    try:
        await logger.info("Starting AutoCare concurrent system...")
        
        # Initialize components in dependency order
        
        # 1. Database connection pool
        await logger.info("Initializing database connection pool...")
        db_pool = ConcurrentDBConnectionPool(settings)
        await db_pool.initialize()
        app.state.db_pool = db_pool
        
        # 2. Session manager (requires Redis)
        await logger.info("Initializing session manager...")
        session_manager = ConcurrentSessionManager()
        await session_manager.initialize()
        app.state.session_manager = session_manager
        
        # 3. WebSocket manager
        await logger.info("Initializing WebSocket manager...")
        websocket_manager = ConcurrentWebSocketManager()
        await websocket_manager.initialize()
        app.state.websocket_manager = websocket_manager
        
        # 4. Agent pool
        await logger.info("Initializing agent pool...")
        agent_pool = ConcurrentAgentPool(settings.get_agent_pool_config())
        await agent_pool.initialize()
        app.state.agent_pool = agent_pool
        
        # 5. Chat service (requires all above components)
        await logger.info("Initializing concurrent chat service...")
        chat_service = ConcurrentChatService(
            session_manager=session_manager,
            websocket_manager=websocket_manager
        )
        await chat_service.initialize()
        app.state.chat_service = chat_service
        
        # Run startup checks
        checker = StartupChecker(logger)
        await checker.run_all()
        
        await logger.info("All concurrent components initialized successfully!")
        
        yield
        
    finally:
        # Cleanup in reverse order
        await logger.info("Shutting down concurrent system...")
        
        try:
            if chat_service:
                await chat_service.close()
                await logger.info("Chat service closed")
            
            if agent_pool:
                await agent_pool.shutdown()
                await logger.info("Agent pool closed")
            
            if websocket_manager:
                await websocket_manager.close()
                await logger.info("WebSocket manager closed")
            
            if session_manager:
                await session_manager.close()
                await logger.info("Session manager closed")
            
            if db_pool:
                await db_pool.close()
                await logger.info("Database connection pool closed")
                
        except Exception as e:
            await logger.error(f"Error during shutdown: {str(e)}")


app = FastAPI(
    title="AutoCare API - Concurrent Edition",
    description="""
    AutoCare API optimized for multiple concurrent users with advanced session management,
    connection pooling, and distributed caching.
    
    Enhanced Features:
    - Concurrent user session management with Redis storage
    - Database connection pooling (10-100 connections)
    - WebSocket connection management with user isolation
    - Agent pool with load balancing
    - Distributed caching and local cache optimization
    - Real-time chat with concurrent user support
    - Advanced rate limiting and security
    - Comprehensive health monitoring and metrics
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc", 
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure security-first CORS
allowed_origins = settings.get_allowed_origins()

# For development, allow all origins if in debug mode
if settings.is_development() or settings.DEBUG:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language", 
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-CSRF-Token",
        "X-User-ID",
        "X-Session-ID",
        "Origin",
        "Cache-Control",
        "X-Requested-With",
    ],
    expose_headers=["X-Total-Count", "X-Rate-Limit-Remaining", "X-Session-ID"],
    max_age=settings.CORS_MAX_AGE,
)

# Add trusted host middleware for additional security
trusted_hosts = ["localhost", "127.0.0.1", "*.yourdomain.com"]
if not settings.is_production():
    trusted_hosts.extend(["*"])  # Allow all hosts in development

app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

# Add security headers middleware
if settings.SECURITY_HEADERS_ENABLED:
    app.add_middleware(SecurityHeadersMiddleware)

app.state.limiter = limiter

# Add security middleware (order matters!)
app.add_middleware(SlowAPIMiddleware)  # Rate limiting

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all incoming requests for debugging"""
    start_time = datetime.now()
    
    # Log basic request info
    await logger.info(f"Request: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    await logger.info(f"Headers: {dict(request.headers)}")
    
    response = await call_next(request)
    
    # Log response
    duration = (datetime.now() - start_time).total_seconds()
    await logger.info(f"Response: {response.status_code} in {duration:.3f}s")
    
    return response


@app.middleware("http")
async def enhanced_security_middleware(request: Request, call_next):
    """Enhanced security middleware stack for concurrent users"""
    try:
        # 1. Input validation
        if not await input_validator.validate_request(request):
            await audit_logger.log_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                ip_address=request.client.host if request.client else None,
                endpoint=request.url.path,
                method=request.method,
                risk_level="high",
                details={"reason": "Invalid or malicious input detected"}
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid or potentially malicious input detected"
            )
        
        # 2. Session context (if available)
        user_id = request.headers.get("X-User-ID")
        session_id = request.headers.get("X-Session-ID")
        
        if user_id and session_manager:
            # Validate session for authenticated requests
            if session_id:
                session_info = await session_manager.get_session(session_id)
                if session_info and session_info.user_id == user_id:
                    request.state.user_id = user_id
                    request.state.session_id = session_id
                    request.state.session_info = session_info
            else:
                # User ID provided but no session ID - this is ok for some endpoints
                request.state.user_id = user_id
        
        # 3. Rate limiting and audit logging
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
        await logger.error(f"Middleware error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


class ErrorResponse(BaseModel):
    detail: str
    code: int
    correlation_id: str = None

# Enhanced error handlers with structured error responses
@app.exception_handler(BaseAPIException)
async def api_exception_handler(request: Request, exc: BaseAPIException):
    """Handle custom API exceptions"""
    correlation_id = getattr(request.state, 'correlation_id', None)
    
    await audit_logger.log_event(
        event_type=AuditEventType.API_REQUEST,
        ip_address=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        status_code=exc.status_code,
        risk_level="medium" if exc.status_code >= 400 else "low",
        details={
            "error_code": exc.error_code.value,
            "correlation_id": exc.correlation_id,
            "user_id": getattr(request.state, 'user_id', None),
            "session_id": getattr(request.state, 'session_id', None)
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
    correlation_id = getattr(request.state, 'correlation_id', None)
    
    await audit_logger.log_event(
        event_type=AuditEventType.API_REQUEST,
        ip_address=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        status_code=exc.status_code,
        risk_level="medium" if exc.status_code >= 400 else "low",
        details={
            "error": exc.detail,
            "correlation_id": correlation_id,
            "user_id": getattr(request.state, 'user_id', None),
            "session_id": getattr(request.state, 'session_id', None)
        }
    )
    
    await logger.error(f"HTTPException: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "status_code": exc.status_code,
            "correlation_id": correlation_id
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
            "retry_after": "60",
            "user_id": getattr(request.state, 'user_id', None)
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
            "correlation_id": correlation_id,
            "user_id": getattr(request.state, 'user_id', None),
            "session_id": getattr(request.state, 'session_id', None)
        }
    )
    
    await logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "SYS_001",
            "message": "Internal server error",
            "correlation_id": correlation_id,
            "status_code": 500
        }
    )


# Dependency injection for concurrent components
async def get_logger_dep() -> ILogger:
    return logger


async def get_websocket_manager_dep() -> ConcurrentWebSocketManager:
    return websocket_manager


async def get_db_pool_dep() -> ConcurrentDBConnectionPool:
    return db_pool


async def get_session_manager_dep() -> ConcurrentSessionManager:
    return session_manager


async def get_chat_service_dep() -> ConcurrentChatService:
    return chat_service


async def get_agent_pool_dep() -> ConcurrentAgentPool:
    return agent_pool


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
        await self.logger.info("All external services connected successfully.")

# Enhanced WebSocket handler for concurrent users
class ConcurrentWebSocketHandler:
    """
    Enhanced WebSocket handler optimized for concurrent users with 
    session management and real-time chat integration.
    """
    def __init__(self, manager: ConcurrentWebSocketManager, logger: ILogger,
                 session_manager: ConcurrentSessionManager, chat_service: ConcurrentChatService):
        self.manager = manager
        self.logger = logger
        self.session_manager = session_manager
        self.chat_service = chat_service

    async def handle(self, websocket: WebSocket, user_id: str = None) -> None:
        connection_info = await self.manager.connect(websocket, user_id)
        connection_id = connection_info.get("connection_id")
        
        try:
            # Send enhanced welcome message
            await self.manager.send_info(
                websocket, 
                "WebSocket connected - concurrent chat enabled", 
                MessageSource.SYSTEM,
                details={
                    "connection_id": connection_id,
                    "user_id": user_id,
                    "features": ["concurrent_chat", "session_management", "real_time_diagnostics"],
                    "connection_time": datetime.now().isoformat(),
                    "max_connections_per_user": settings.WEBSOCKET_MAX_CONNECTIONS_PER_USER
                }
            )
            
            while True:
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    message_type = message.get("type", "unknown")
                    
                    if message_type == "chat":
                        await self._handle_chat_message(websocket, message, user_id)
                    elif message_type == "ping":
                        await self.manager.send_info(
                            websocket, "pong", MessageSource.SYSTEM,
                            details={"timestamp": datetime.now().isoformat()}
                        )
                    elif message_type == "session_info":
                        await self._handle_session_info_request(websocket, message, user_id)
                    elif message_type == "test":
                        await self.manager.send_info(
                            websocket, 
                            f"Test message received: {message.get('data', '')}", 
                            MessageSource.SYSTEM,
                            details={"original_message": message, "echo_time": datetime.now().isoformat()}
                        )
                    else:
                        # Echo back unknown message types with warning
                        await self.manager.send_info(
                            websocket,
                            f"Unknown message type: {message_type}",
                            MessageSource.SYSTEM,
                            details={"supported_types": ["chat", "ping", "session_info", "test"]}
                        )
                        
                except json.JSONDecodeError:
                    # Handle plain text messages
                    await self.manager.send_personal_message(
                        f"Received plain text: {data}", websocket
                    )
                    
        except WebSocketDisconnect:
            await self.manager.disconnect(websocket)
            await self.logger.info(f"WebSocket disconnected for user: {user_id}")
        except Exception as exc:
            await self.logger.error(f"WebSocket error for user {user_id}: {str(exc)}")
            await self.manager.disconnect(websocket)
            await websocket.close(code=1011)
    
    async def _handle_chat_message(self, websocket: WebSocket, message: dict, user_id: str):
        """Handle chat messages through the concurrent chat service"""
        try:
            session_id = message.get("session_id")
            content = message.get("message", "")
            context = message.get("context", {})
            
            if not content:
                await self.manager.send_error(
                    websocket, 
                    "Missing required field: message", 
                    MessageSource.CHAT_SERVICE
                )
                return
            
            # Send processing notification
            await self.manager.send_stage(
                websocket, 
                "Processing chat message", 
                MessageSource.CHAT_SERVICE,
                session_id=session_id
            )
            
            # Process through concurrent chat service
            response = await self.chat_service.send_message(
                user_id=user_id or "anonymous",
                message=content,
                context=context,
                session={"id": session_id} if session_id else None,
                websocket=websocket
            )
            
            # Send final response
            await self.manager.send_result(
                websocket, 
                "Chat message processed", 
                MessageSource.CHAT_SERVICE,
                session_id=response.get("session_id"),
                details=response
            )
            
        except Exception as e:
            await self.logger.error(f"Error handling chat message: {str(e)}")
            await self.manager.send_error(
                websocket, 
                f"Error processing chat message: {str(e)}", 
                MessageSource.CHAT_SERVICE,
                session_id=message.get("session_id")
            )
    
    async def _handle_session_info_request(self, websocket: WebSocket, message: dict, user_id: str):
        """Handle session information requests"""
        try:
            if not user_id:
                await self.manager.send_error(
                    websocket,
                    "User ID required for session info",
                    MessageSource.SYSTEM
                )
                return
            
            session_id = message.get("session_id")
            session_info = await self.session_manager.get_session_info(user_id, session_id)
            
            await self.manager.send_result(
                websocket,
                "Session information retrieved",
                MessageSource.SYSTEM,
                details=session_info
            )
            
        except Exception as e:
            await self.logger.error(f"Error retrieving session info: {str(e)}")
            await self.manager.send_error(
                websocket,
                f"Error retrieving session info: {str(e)}",
                MessageSource.SYSTEM
            )

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
    description="Welcome endpoint for the concurrent AutoCare API.",
)
@limiter.limit("20/minute")
async def read_root(request: Request):
    return {
        "message": "Welcome to AutoCare API - Concurrent Edition",
        "version": "2.0.0",
        "features": [
            "concurrent_users",
            "session_management", 
            "connection_pooling",
            "agent_load_balancing",
            "distributed_caching"
        ]
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="System Health Check",
    description="Comprehensive health check for all concurrent system components.",
)
async def health_check():
    """Enhanced health check for concurrent system"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": {}
    }
    
    try:
        # Check database connection pool
        if db_pool:
            pool_stats = await db_pool.get_stats()
            health_status["components"]["database"] = {
                "status": "healthy",
                "active_connections": pool_stats.get("active_connections", 0),
                "max_connections": pool_stats.get("max_connections", 0),
                "pool_utilization": f"{pool_stats.get('utilization', 0):.1f}%"
            }
        
        # Check session manager
        if session_manager:
            session_stats = await session_manager.get_stats()
            health_status["components"]["session_manager"] = {
                "status": "healthy",
                "active_sessions": session_stats.get("active_sessions", 0),
                "total_users": session_stats.get("total_users", 0)
            }
        
        # Check WebSocket manager
        if websocket_manager:
            ws_stats = await websocket_manager.get_connection_stats()
            health_status["components"]["websocket"] = {
                "status": "healthy",
                "active_connections": ws_stats.get("active_connections", 0),
                "total_users": ws_stats.get("total_users", 0)
            }
        
        # Check agent pool
        if agent_pool:
            agent_stats = await agent_pool.get_pool_stats()
            health_status["components"]["agent_pool"] = {
                "status": "healthy",
                "total_agents": agent_stats.get("total_agents", 0),
                "active_agents": agent_stats.get("active_agents", 0),
                "queue_size": agent_stats.get("queue_size", 0)
            }
        
        # Check Redis
        try:
            redis_cache = await get_redis_cache()
            is_healthy = await redis_cache.health_check()
            if is_healthy:
                health_status["components"]["redis"] = {"status": "healthy"}
            else:
                health_status["components"]["redis"] = {"status": "unhealthy", "error": "Redis health check failed"}
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "degraded"
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status


@app.get(
    "/health/database",
    tags=["Health"],
    summary="Database Health Check",
    description="Check database connection pool health.",
)
async def health_check_database():
    """Database-specific health check"""
    try:
        if db_pool:
            pool_stats = await db_pool.get_stats()
            return {
                "status": "healthy",
                "active_connections": pool_stats.get("active_connections", 0),
                "max_connections": pool_stats.get("max_connections", 0),
                "pool_utilization": f"{pool_stats.get('utilization', 0):.1f}%"
            }
        else:
            return {"status": "unhealthy", "error": "Database pool not initialized"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get(
    "/health/redis",
    tags=["Health"],
    summary="Redis Health Check",
    description="Check Redis connection health.",
)
async def health_check_redis():
    """Redis-specific health check"""
    try:
        redis_cache = await get_redis_cache()
        is_healthy = await redis_cache.health_check()
        if is_healthy:
            return {"status": "healthy"}
        else:
            return {"status": "unhealthy", "error": "Redis ping failed"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get(
    "/health/agents",
    tags=["Health"],
    summary="Agent Pool Health Check",
    description="Check agent pool health and statistics.",
)
async def health_check_agents():
    """Agent pool-specific health check"""
    try:
        if agent_pool:
            agent_stats = await agent_pool.get_pool_stats()
            return {
                "status": "healthy",
                "total_agents": agent_stats.get("total_agents", 0),
                "active_agents": agent_stats.get("active_agents", 0),
                "queue_size": agent_stats.get("queue_size", 0)
            }
        else:
            return {"status": "unhealthy", "error": "Agent pool not initialized"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get(
    "/health/sessions",
    tags=["Health"],
    summary="Session Manager Health Check",
    description="Check session manager health and statistics.",
)
async def health_check_sessions():
    """Session manager-specific health check"""
    try:
        if session_manager:
            session_stats = await session_manager.get_stats()
            return {
                "status": "healthy",
                "active_sessions": session_stats.get("active_sessions", 0),
                "total_users": session_stats.get("total_users", 0)
            }
        else:
            return {"status": "unhealthy", "error": "Session manager not initialized"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="System Metrics",
    description="Detailed system metrics for monitoring and performance analysis.",
)
async def system_metrics():
    """Get comprehensive system metrics"""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "version": "2.0.0",
            "uptime": "calculated_at_runtime"  # Could be implemented
        }
    }
    
    try:
        # Database metrics
        if db_pool:
            metrics["database"] = await db_pool.get_stats()
        
        # Session metrics
        if session_manager:
            metrics["sessions"] = await session_manager.get_stats()
        
        # WebSocket metrics
        if websocket_manager:
            metrics["websockets"] = await websocket_manager.get_connection_stats()
        
        # Agent pool metrics
        if agent_pool:
            metrics["agents"] = await agent_pool.get_pool_stats()
        
        # Chat service metrics
        if chat_service:
            # Chat service may not have get_stats method, let's skip it for now
            metrics["chat"] = {"status": "active"}
            
    except Exception as e:
        metrics["error"] = str(e)
    
    return metrics


@app.websocket(
    "/ws",
    name="Enhanced WebSocket Endpoint",
)
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = None,
    websocket_manager_dep: ConcurrentWebSocketManager = Depends(get_websocket_manager_dep),
    logger_dep: ILogger = Depends(get_logger_dep),
    session_manager_dep: ConcurrentSessionManager = Depends(get_session_manager_dep),
    chat_service_dep: ConcurrentChatService = Depends(get_chat_service_dep),
):
    """
    Enhanced WebSocket endpoint optimized for concurrent users.
    Supports user-based connection tracking, session management,
    and real-time chat integration.
    """
    handler = ConcurrentWebSocketHandler(
        websocket_manager_dep, logger_dep, session_manager_dep, chat_service_dep
    )
    await handler.handle(websocket, user_id)
