"""Keeps signed download urls out of uvicorn's access log."""
import logging
import re

_SIGNED = re.compile(r"([?&](?:exp|sig)=)[^&\s\"']*")


def _redact(value):
    return _SIGNED.sub(r"\1REDACTED", value) if isinstance(value, str) else value


class RedactSignedUrls(logging.Filter):
    """Replaces the value of exp= and sig= query parameters in a log record; never raises, never drops a record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = _redact(record.msg)
            if isinstance(record.args, tuple):
                record.args = tuple(_redact(a) for a in record.args)
        except Exception:
            pass
        return True


def install_redaction() -> None:
    """Idempotent: attach one RedactSignedUrls to the uvicorn.access logger."""
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, RedactSignedUrls) for f in access.filters):
        access.addFilter(RedactSignedUrls())
