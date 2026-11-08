import logging

from app.utils.log_redaction import RedactSignedUrls, install_redaction

SHA = "a" * 64
PATH = f"/api/v1/visuals/files/{SHA}.webp"


def _emit(args, msg='%s - "%s %s HTTP/%s" %d'):
    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, msg, args, None)
    assert RedactSignedUrls().filter(record) is True
    return record


def test_signed_params_are_redacted():
    r = _emit(("127.0.0.1:1", "GET", f"{PATH}?exp=123&sig={SHA}", "1.1", 200))
    assert r.getMessage() == f'127.0.0.1:1 - "GET {PATH}?exp=REDACTED&sig=REDACTED HTTP/1.1" 200'


def test_redacts_when_params_are_in_the_message_itself():
    r = _emit((), msg=f"GET {PATH}?sig={SHA}&exp=9 done")
    assert r.getMessage() == f"GET {PATH}?sig=REDACTED&exp=REDACTED done"


def test_other_records_are_unchanged():
    r = _emit(("127.0.0.1:1", "GET", "/api/v1/cars?limit=5", "1.1", 200))
    assert r.getMessage() == '127.0.0.1:1 - "GET /api/v1/cars?limit=5 HTTP/1.1" 200'


def test_odd_args_pass_through_without_raising():
    assert _emit(None, msg="plain").getMessage() == "plain"
    assert _emit(({"k": 1},), msg="%(k)s").getMessage() == "1"
    assert _emit((None, 5, b"sig=x", 200), msg="%s %s %s %s").args[0] is None
    record = logging.LogRecord("x", logging.INFO, __file__, 1, ("not", "a str"), None, None)
    assert RedactSignedUrls().filter(record) is True


def test_install_is_idempotent():
    access = logging.getLogger("uvicorn.access")
    install_redaction()
    install_redaction()
    assert sum(isinstance(f, RedactSignedUrls) for f in access.filters) == 1
