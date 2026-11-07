"""Every error response has one body shape: {"error_code", "message", "correlation_id", "timestamp", ...}."""
import pytest
from fastapi.testclient import TestClient

from app.utils.exceptions import http_error_body

KEYS = {"error_code", "message", "correlation_id", "timestamp"}


def _error(r, status: int, code: str, message: str | None = None) -> dict:
    body = r.json()
    assert r.status_code == status, body
    assert KEYS <= body.keys() and "detail" not in body
    assert all(isinstance(body[k], str) for k in KEYS)
    assert body["error_code"] == code
    if message is not None:
        assert body["message"] == message
    return body


def test_unknown_route_is_404_in_the_error_shape(client):
    _error(client.get("/api/v1/nope"), 404, "SYS_004", "Not Found")


def test_wrong_method_is_405_in_the_error_shape_and_keeps_allow(client):
    r = client.put("/health")
    _error(r, 405, "SYS_005", "Method Not Allowed")
    assert "GET" in r.headers["allow"]


def test_plain_http_exception_detail_becomes_the_message(client):
    _error(client.post("/api/v1/auth/google", json={"id_token": "t"}), 404, "SYS_004", "Not Found")  # sign-in off


def test_missing_token_401_keeps_the_bearer_challenge(client):
    r = client.get("/api/v1/cars")
    _error(r, 401, "AUTH_001", "Missing authentication token")
    assert r.headers["www-authenticate"] == "Bearer"


def test_api_exception_keeps_its_code(client, auth_headers, mocker):
    mocker.patch("app.api.v1.car_route.user_profile_service.remove_car_from_user", return_value={"success": False})
    _error(client.delete("/api/v1/cars/toyota-echo-2001", headers=auth_headers), 404, "DB_002", "Car not found")


def test_validation_422_names_the_fields_but_never_echoes_the_input(client, auth_headers):
    secret = "Toyota\x00" + "s3cret-value-xyz"
    r = client.post("/api/v1/cars", json={"make": secret, "model": "Echo", "year": 99999999}, headers=auth_headers)
    body = _error(r, 422, "VAL_001", "Invalid request")
    assert body["fields"] == ["make", "year"]
    assert "s3cret-value-xyz" not in r.text and "99999999" not in r.text


def test_validation_422_for_a_bad_email_and_missing_field(client):
    r = client.post("/api/v1/auth/login", json={"email": "not-an-email-s3cret"})
    body = _error(r, 422, "VAL_001")
    assert body["fields"] == ["email", "password"] and "not-an-email-s3cret" not in r.text


def test_validation_422_for_unparsable_json_and_query_or_path_values(client, auth_headers):
    r = client.post(
        "/api/v1/cars", content=b'{"make": "s3cret', headers={**auth_headers, "content-type": "application/json"}
    )
    assert _error(r, 422, "VAL_001")["fields"] == ["body"] and "s3cret" not in r.text
    r = client.delete("/api/v1/cars/" + "x" * 300, headers=auth_headers)
    assert _error(r, 422, "VAL_001")["fields"] == ["car_id"] and "x" * 300 not in r.text


def test_body_too_large_is_413_in_the_error_shape(client):
    r = client.post("/api/v1/auth/login", content=b"x" * (1024 * 1024 + 1), headers={"content-type": "application/json"})
    _error(r, 413, "VAL_005", "Request body too large")


def test_rate_limit_is_429_in_the_error_shape(client, fake_users):
    for _ in range(10):
        client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "s3cret-pass"})
    r = client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "s3cret-pass"})
    body = _error(r, 429, "EXT_003")
    assert body["message"] == "Rate limit exceeded: 10 per 1 minute"


def test_unhandled_error_is_500_in_the_error_shape(auth_headers, mocker):
    import main

    secret = "pg: password=hunter2"
    mocker.patch("app.services.diagnosis.sessions.list_for_user", side_effect=RuntimeError(secret))
    with TestClient(main.app, raise_server_exceptions=False) as c:
        r = c.get("/api/v1/sessions", headers=auth_headers)
    _error(r, 500, "SYS_001", "Internal server error")
    assert secret not in r.text


@pytest.mark.parametrize(
    "status, detail, code, message",
    [
        (415, "Unsupported file type.", "VAL_003", "Unsupported file type."),
        (413, "File is too large.", "VAL_005", "File is too large."),
        (422, "Send only one audio or video file.", "VAL_001", "Send only one audio or video file."),
        (418, None, "HTTP_418", "I'm a Teapot"),
    ],
)
def test_http_error_body_maps_plain_statuses(status, detail, code, message):
    body = http_error_body(status, detail)
    assert KEYS <= body.keys() and body["error_code"] == code and body["message"] == message


def test_http_error_body_passes_api_exception_bodies_through():
    body = {"error_code": "DB_002", "message": "Car not found", "correlation_id": "c", "timestamp": "t"}
    assert http_error_body(404, body) is body


# --- database outage: 503 in every route family, never "not found" or an empty list ----------------

SID = "33333333-3333-4333-8333-333333333333"
LEAK = "password=hunter2 host=10.0.0.5"


def _unavailable(r):
    _error(r, 503, "SYS_002", "Service temporarily unavailable")
    assert LEAK not in r.text


@pytest.fixture
def db_down(fake_pool):
    fake_pool.error = ConnectionRefusedError(LEAK)
    return fake_pool


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("GET", "/api/v1/cars", None),
        ("POST", "/api/v1/cars", {"make": "Toyota", "model": "Echo", "year": 2001}),
        ("DELETE", "/api/v1/cars/toyota-echo-2001", None),
    ],
)
def test_cars_answer_503_during_a_database_outage(client, auth_headers, db_down, method, path, body):
    _unavailable(client.request(method, path, json=body, headers=auth_headers))


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("POST", "/api/v1/sessions", {"car_id": "toyota-echo-2001"}),  # was 404 "Car not found"
        ("GET", "/api/v1/sessions", None),
        ("GET", f"/api/v1/sessions/{SID}", None),
        ("DELETE", f"/api/v1/sessions/{SID}", None),
    ],
)
def test_sessions_answer_503_during_a_database_outage(client, auth_headers, db_down, method, path, body):
    _unavailable(client.request(method, path, json=body, headers=auth_headers))


def test_sending_a_message_answers_503_during_a_database_outage(client, auth_headers, db_down):
    _unavailable(client.post(f"/api/v1/sessions/{SID}/messages", data={"text": "fan clicks"}, headers=auth_headers))


@pytest.mark.parametrize(
    "path, body",
    [
        ("/api/v1/auth/register", {"email": "driver@example.com", "password": "s3cret-pass"}),
        ("/api/v1/auth/login", {"email": "driver@example.com", "password": "s3cret-pass"}),
    ],
)
def test_sign_in_answers_503_during_a_database_outage(client, db_down, path, body):
    _unavailable(client.post(path, json=body))


def test_google_sign_in_answers_503_during_a_database_outage(client, db_down, mocker, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "GOOGLE_CLIENT_IDS", "client-1")
    mocker.patch("google.oauth2.id_token.verify_oauth2_token", return_value={
        "iss": "accounts.google.com", "aud": "client-1", "sub": "s", "email": "d@gmail.com", "email_verified": True,
    })
    _unavailable(client.post("/api/v1/auth/google", json={"id_token": "t"}))


def test_token_checks_answer_503_during_a_database_outage(client, db_down):
    from app.utils.auth_middleware import jwt_handler

    token = jwt_handler.create_access_token({"sub": "11111111-1111-4111-8111-111111111111"})
    _unavailable(client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"}))
    _unavailable(client.get("/api/v1/cars", headers={"Authorization": f"Bearer {token}"}))
    refresh = jwt_handler.create_refresh_token({"sub": "11111111-1111-4111-8111-111111111111"})
    _unavailable(client.post("/api/v1/auth/refresh", json={"refresh_token": refresh}))
