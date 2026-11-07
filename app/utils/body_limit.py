"""Reject oversized request bodies before the app reads them (pure ASGI, so nothing is spooled first)."""
import re

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.utils.correlation import correlation_id
from app.utils.exceptions import PayloadTooLargeException


class BodySizeLimitMiddleware:
    """413 when the declared Content-Length is over the cap, or when a body without one (chunked) grows past it.

    Paths matching `upload_path` get `upload_max_bytes`; every other request gets `max_bytes`.
    """

    def __init__(self, app: ASGIApp, max_bytes: int, upload_max_bytes: int, upload_path: str):
        self.app = app
        self.max_bytes = max_bytes
        self.upload_max_bytes = upload_max_bytes
        self.upload_path = re.compile(upload_path)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.upload_max_bytes if self.upload_path.fullmatch(scope["path"]) else self.max_bytes

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            await _reject(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # An HTTPException from receive is re-raised by FastAPI's body parsing and becomes this 413.
                    raise PayloadTooLargeException()
            return message

        await self.app(scope, limited_receive, send)


async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
    exc = PayloadTooLargeException()
    body = {**exc.detail, "correlation_id": correlation_id(scope) or exc.correlation_id}
    await JSONResponse(status_code=exc.status_code, content=body)(scope, receive, send)
