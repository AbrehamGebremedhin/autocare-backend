"""AutoCare backend entrypoint."""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.v1.routes import router as v1_router
from app.core.config import get_settings
from app.db import pool
from app.services import llm
from app.utils.body_limit import BodySizeLimitMiddleware
from app.utils.exceptions import BaseAPIException
from app.utils.limiter import limiter
from app.utils.security_middleware import SecurityHeadersMiddleware

settings = get_settings()
logger = logging.getLogger("autocare")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.init_pool()
    try:
        yield
    finally:
        await pool.close_pool()


app = FastAPI(title="AutoCare API", debug=settings.DEBUG, lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
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


@app.exception_handler(BaseAPIException)
async def api_exception_handler(request: Request, exc: BaseAPIException):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log type and route only; no request data, headers or stack trace in the response.
    logger.error("Unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error_code": "SYS_001", "message": "Internal server error"},
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
    return {"status": "ok", "database": database, "deepseek": deepseek}
