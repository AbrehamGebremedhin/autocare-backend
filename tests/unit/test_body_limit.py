"""Request body cap: 413 before the body is read when Content-Length is too big, and while reading when it is absent."""
import json
import time

import pytest
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.testclient import TestClient

from app.utils.auth_middleware import access_claims, jwt_handler
from app.utils.body_limit import BodySizeLimitMiddleware
from app.utils.exceptions import BaseAPIException

UPLOAD = "/upload/x"


def _app(reads: list) -> FastAPI:
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=10, upload_max_bytes=100, upload_path=r"/upload/[^/]+")

    @app.exception_handler(BaseAPIException)
    async def handler(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    @app.post("/json")
    @app.post(UPLOAD)
    async def echo(request: Request):
        body = await request.body()
        reads.append(len(body))
        return {"n": len(body)}

    @app.post("/upload/form")
    async def form(files: list[UploadFile] = File(...)):
        reads.append(sum([f.size or 0 for f in files]))
        return {}

    return app


async def _call(app, path: str, headers: list, chunks: list[bytes]) -> tuple[int, list[dict], int]:
    """Drive the ASGI app directly; returns (status, sent messages, number of receive calls)."""
    calls = 0
    sent: list[dict] = []

    async def receive():
        nonlocal calls
        calls += 1
        if calls <= len(chunks):
            return {"type": "http.request", "body": chunks[calls - 1], "more_body": calls < len(chunks)}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "headers": headers, "client": ("127.0.0.1", 1), "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], sent, calls


async def test_declared_length_over_cap_is_413_without_reading_body():
    reads: list = []
    status, _, calls = await _call(_app(reads), "/json", [(b"content-length", b"11")], [b"x" * 11])
    assert status == 413
    assert calls == 0 and reads == []


async def test_upload_path_gets_the_larger_cap():
    reads: list = []
    app = _app(reads)
    assert (await _call(app, UPLOAD, [(b"content-length", b"100")], [b"x" * 100]))[0] == 200
    status, _, calls = await _call(app, UPLOAD, [(b"content-length", b"101")], [b"x" * 101])
    assert status == 413 and calls == 0
    assert reads == [100]


async def test_malformed_content_length_is_rejected():
    status, _, calls = await _call(_app([]), "/json", [(b"content-length", b"12abc")], [b"x"])
    assert status == 413 and calls == 0


async def test_chunked_body_is_cut_off_once_it_passes_the_cap():
    reads: list = []
    status, sent, calls = await _call(_app(reads), "/json", [(b"transfer-encoding", b"chunked")], [b"x" * 6] * 5)
    assert status == 413
    assert calls == 2  # stopped at the chunk that crossed 10 bytes, the rest was never pulled
    assert reads == []
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert b"VAL_005" in body


def test_chunked_body_through_test_client():
    reads: list = []
    with TestClient(_app(reads)) as c:
        r = c.post("/json", content=iter([b"x" * 6, b"x" * 6]))
        assert r.status_code == 413
        assert c.post("/json", content=iter([b"x" * 4, b"x" * 4])).status_code == 200
    assert reads == [8]


def test_chunked_multipart_upload_is_cut_off_during_form_parsing():
    reads: list = []
    part = (
        b'--b\r\nContent-Disposition: form-data; name="files"; filename="a.bin"\r\n\r\n'
        + b"x" * 200
        + b"\r\n--b--\r\n"
    )
    with TestClient(_app(reads)) as c:
        r = c.post("/upload/form", content=iter([part[:60], part[60:]]),
                   headers={"content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413
    assert reads == []


def test_small_bodies_pass():
    reads: list = []
    with TestClient(_app(reads)) as c:
        assert c.post("/json", content=b"x" * 10).json() == {"n": 10}


@pytest.fixture
def main_app(client):
    return client.app


def test_main_app_caps_json_routes_at_1mb(client):
    r = client.post("/api/v1/auth/login", content=b"{" + b" " * (1024 * 1024) + b"}",
                    headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert r.json()["error_code"] == "VAL_005"


async def test_main_app_rejects_huge_upload_before_auth_and_body_read(main_app):
    from app.core.config import get_settings

    s = get_settings()
    cap = (max(s.MAX_IMAGES * s.MAX_IMAGE_MB, s.MAX_MEDIA_MB) + 1) * 1024 * 1024
    path = "/api/v1/sessions/abc/messages"
    status, _, calls = await _call(main_app, path, [(b"content-length", str(cap + 1).encode())], [b""])
    assert status == 413 and calls == 0  # no token was sent: the cap runs before auth


def _auth_app(reads: list) -> FastAPI:
    app = _app(reads)
    app.user_middleware.clear()
    app.middleware_stack = None
    app.add_middleware(
        BodySizeLimitMiddleware, max_bytes=10, upload_max_bytes=100, upload_path=r"/upload/[^/]+", upload_auth=access_claims
    )
    return app


def _json(sent: list[dict]) -> dict:
    return json.loads(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))


def _bearer(token: str, scheme: bytes = b"Bearer") -> list:
    return [(b"authorization", scheme + b" " + token.encode())]


async def test_upload_without_a_token_is_401_and_the_body_is_never_read():
    status, sent, receives = await _call(_auth_app([]), UPLOAD, [], [b"x" * 50])
    assert status == 401 and receives == 0
    assert _json(sent)["error_code"] == "AUTH_001"
    assert _json(sent)["message"] == "Missing authentication token"
    assert (b"www-authenticate", b"Bearer") in sent[0]["headers"]


async def test_upload_with_an_expired_token_says_so():
    token = jwt_handler.create_access_token({"sub": "u1"}, issued_at=time.time() - 7200)
    status, sent, receives = await _call(_auth_app([]), UPLOAD, _bearer(token), [b"x" * 50])
    assert status == 401 and receives == 0
    assert _json(sent)["message"] == "Token has expired"


async def test_upload_with_a_refresh_token_is_401():
    token = jwt_handler.create_refresh_token({"sub": "u1"})
    status, sent, receives = await _call(_auth_app([]), UPLOAD, _bearer(token), [b"x" * 50])
    assert status == 401 and receives == 0
    assert _json(sent)["message"] == "Invalid token type"


async def test_upload_accepts_a_lower_case_bearer_scheme():
    reads: list = []
    token = jwt_handler.create_access_token({"sub": "u1"})
    status, _, _ = await _call(_auth_app(reads), UPLOAD, _bearer(token, b"bearer"), [b"x" * 50])
    assert status == 200 and reads == [50]


async def test_oversized_upload_is_still_413_without_a_token():
    status, _, receives = await _call(_auth_app([]), UPLOAD, [(b"content-length", b"101")], [b"x" * 101])
    assert status == 413 and receives == 0


async def test_json_routes_are_not_checked():
    reads: list = []
    status, _, _ = await _call(_auth_app(reads), "/json", [], [b"x" * 5])
    assert status == 200 and reads == [5]


def test_main_app_checks_the_token_of_chat_uploads_before_reading_them():
    import main

    limits = [m for m in main.app.user_middleware if m.cls is BodySizeLimitMiddleware]
    assert [m.kwargs["upload_auth"] for m in limits] == [access_claims]
