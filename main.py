"""AutoCare backend entrypoint."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.routes import router as v1_router
from app.core.config import get_settings
from app.db import pool
from app.db.pool import DatabaseUnavailable
from app.services import llm
from app.utils.body_limit import BodySizeLimitMiddleware
from app.utils.exceptions import ErrorCode, ServiceUnavailableException, error_body, http_error_body
from app.utils.limiter import limiter
from app.utils.security_middleware import SecurityHeadersMiddleware

settings = get_settings()
logger = logging.getLogger("autocare")


def media_dir_writable(path: str) -> bool:
    """True if uploads can be stored: MEDIA_DIR, or its nearest existing parent (uploads create the rest), is writable."""
    current = Path(path).absolute()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current.is_dir() and os.access(current, os.W_OK | os.X_OK)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not media_dir_writable(settings.MEDIA_DIR):
        logger.warning("MEDIA_DIR %s is not writable by this process; media uploads will fail", settings.MEDIA_DIR)
    await pool.init_pool()
    try:
        yield
    finally:
        await pool.close_pool()


app = FastAPI(title="AutoCare API", debug=settings.DEBUG, lifespan=lifespan)

app.state.limiter = limiter
# Bodies are capped before FastAPI reads them (multipart files would otherwise be spooled to disk before auth).
# Only the chat route takes uploads: the largest legal payload plus 1 MB for the form fields. JSON routes get 1 MB.
_MB = 1024 * 1024
app.add_middleware(
    BodySizeLimitMiddleware,
    max_bytes=_MB,
    upload_max_bytes=(max(settings.MAX_IMAGES * settings.MAX_IMAGE_MB, settings.MAX_MEDIA_MB) + 1) * _MB,
    upload_path=r"/api/v1/sessions/[^/]+/messages",
)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Every error response has the same body: {"error_code", "message", "correlation_id", "timestamp", ...}
# (app.utils.exceptions.error_body). None of them echoes request data.

MAX_REPORTED_FIELDS = 20
_SOURCES = {"body", "query", "path", "header", "cookie"}


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    # BaseAPIException, FastAPI's and Starlette's HTTPExceptions (unknown route 404, wrong method 405), media 413/415.
    if exc.status_code in (204, 304):
        return Response(status_code=exc.status_code, headers=exc.headers)
    return JSONResponse(
        status_code=exc.status_code, content=http_error_body(exc.status_code, exc.detail), headers=exc.headers
    )


def invalid_fields(errors) -> list[str]:
    """The names of the invalid fields ("make", "files"), without the request source, list indexes or the
    rejected values themselves."""
    names: list[str] = []
    for error in errors:
        loc = list(error.get("loc", ()))
        source = str(loc.pop(0)) if loc and loc[0] in _SOURCES else "body"
        name = ".".join(part for part in loc if isinstance(part, str))[:64] or source
        if name not in names:
            names.append(name)
    return names[:MAX_REPORTED_FIELDS]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=error_body(ErrorCode.INVALID_INPUT.value, "Invalid request", fields=invalid_fields(exc.errors())),
    )


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    # Sync on purpose: SlowAPIMiddleware calls the registered handler only when it is not a coroutine.
    response = JSONResponse(
        status_code=429, content=error_body(ErrorCode.RATE_LIMIT_EXCEEDED.value, f"Rate limit exceeded: {exc.detail}")
    )
    return request.app.state.limiter._inject_headers(response, request.state.view_rate_limit)  # as slowapi's own


app.add_exception_handler(RateLimitExceeded, rate_limit_handler)


@app.exception_handler(DatabaseUnavailable)
async def database_unavailable_handler(request: Request, exc: DatabaseUnavailable):
    # The same 503 as a token check during an outage: clients keep their tokens and retry.
    logger.error("Database unavailable (%s) on %s %s", exc, request.method, request.url.path)
    unavailable = ServiceUnavailableException()
    return JSONResponse(status_code=unavailable.status_code, content=unavailable.detail)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log type and route only; no request data, headers or stack trace in the response.
    logger.error("Unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=error_body(ErrorCode.INTERNAL_SERVER_ERROR.value, "Internal server error"),
    )


app.include_router(v1_router, prefix="/api/v1")


async def check_database() -> bool:
    try:
        return await asyncio.wait_for(pool.fetchval("select 1"), 5) == 1
    except Exception:
        return False


@app.get("/health")
async def health():
    database, deepseek = await asyncio.gather(check_database(), llm.ping())
    # 503 without the database, so the compose healthcheck reports it; DeepSeek being down alone keeps 200.
    return JSONResponse(
        status_code=200 if database else 503,
        content={"status": "ok" if database else "unavailable", "database": database, "deepseek": deepseek},
    )
