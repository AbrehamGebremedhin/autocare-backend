"""AutoCare backend entrypoint."""
import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.v1.routes import router as v1_router
from app.core.config import get_settings
from app.db.base import SupabaseDBHandler
from app.services import llm
from app.utils.exceptions import BaseAPIException
from app.utils.limiter import limiter
from app.utils.security_middleware import SecurityHeadersMiddleware

settings = get_settings()
logger = logging.getLogger("autocare")

app = FastAPI(title="AutoCare API", debug=settings.DEBUG)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
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


def _ping_supabase() -> None:
    SupabaseDBHandler().client.table("Car").select("id").limit(1).execute()


async def check_supabase() -> bool:
    try:
        await asyncio.wait_for(asyncio.to_thread(_ping_supabase), 5)
        return True
    except Exception:
        return False


@app.get("/health")
async def health():
    supabase, deepseek = await asyncio.gather(check_supabase(), llm.ping())
    return {"status": "ok", "supabase": supabase, "deepseek": deepseek}
