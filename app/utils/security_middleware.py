"""Security headers on every HTTP response (pure ASGI, so SSE tokens are not copied through an extra task and stream)."""
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import get_settings

DOCS_PATHS = {"/docs", "/docs/oauth2-redirect", "/redoc"}


class SecurityHeadersMiddleware:
    """An API answers JSON and SSE only, so the policy is the strictest one. Swagger UI / ReDoc need scripts and
    CDN assets, so only those pages skip the CSP and framing headers (and only when ENABLE_DOCS serves them).
    Nothing is cacheable unless the route sets its own Cache-Control (the SSE stream, signed artwork)."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        settings = get_settings()
        is_docs = settings.ENABLE_DOCS and scope["path"] in DOCS_PATHS

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
                if not is_docs:
                    headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
                    headers["X-Frame-Options"] = "DENY"
                if settings.is_production():
                    headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
                headers.setdefault("Cache-Control", "no-store")
            await send(message)

        await self.app(scope, receive, send_with_headers)
