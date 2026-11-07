"""One correlation id per request, so a client's error report can be matched to the server's log lines.

The id is set as `request.state.correlation_id` (routes and handlers log it as `[ID: ...]`), returned in the
`X-Correlation-ID` header of every response, and used as the `correlation_id` of every error body. It is
always generated here; a client-sent id is not trusted into the logs.
"""
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

HEADER = b"x-correlation-id"


def correlation_id(scope: Scope) -> str | None:
    return scope.get("state", {}).get("correlation_id")


class CorrelationIdMiddleware:
    """Pure ASGI (nothing is buffered, SSE streams pass through untouched). Install it outermost."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        cid = str(uuid4())
        scope.setdefault("state", {})["correlation_id"] = cid

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), (HEADER, cid.encode())]
            await send(message)

        await self.app(scope, receive, send_with_id)
