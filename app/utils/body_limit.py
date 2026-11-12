"""Reject oversized request bodies before the app reads them (pure ASGI, so nothing is spooled first)."""
import re
from collections.abc import Callable

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.utils.correlation import correlation_id
from app.utils.exceptions import PayloadTooLargeException, http_error_body


class BodySizeLimitMiddleware:
    """413 when the declared Content-Length is over the cap, or when a body without one (chunked) grows past it.

    Paths matching `upload_path` get `upload_max_bytes`; every other request gets `max_bytes`. When `upload_auth`
    is given, those paths also need a token it accepts (it gets the bearer token or None and raises an
    HTTPException): FastAPI parses an upload before it runs the route's auth dependency, so without this an
    anonymous client could make the server spool the whole body.
    """

    def __init__(
        self, app: ASGIApp, max_bytes: int, upload_max_bytes: int, upload_path: str,
        upload_auth: Callable[[str | None], object] | None = None,
    ):
        self.app = app
        self.upload_auth = upload_auth
        self.max_bytes = max_bytes
        self.upload_max_bytes = upload_max_bytes
        self.upload_path = re.compile(upload_path)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_upload = self.upload_path.fullmatch(scope["path"]) is not None
        limit = self.upload_max_bytes if is_upload else self.max_bytes

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            await _reject(scope, receive, send, PayloadTooLargeException())
            return

        if is_upload and self.upload_auth is not None:
            try:
                self.upload_auth(_bearer_token(scope))
            except HTTPException as exc:
                await _reject(scope, receive, send, exc)
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


def _bearer_token(scope: Scope) -> str | None:
    """The token of an `Authorization: Bearer <token>` header (any case of the scheme), else None."""
    value = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else None


async def _reject(scope: Scope, receive: Receive, send: Send, exc: HTTPException) -> None:
    body = http_error_body(exc.status_code, exc.detail)
    body = {**body, "correlation_id": correlation_id(scope) or body.get("correlation_id")}
    await JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)(scope, receive, send)
