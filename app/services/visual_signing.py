"""Self-authenticating download urls for vehicle artwork.

The mobile client downloads with a plain GET and no Authorization header, so the url carries an expiry and an
HMAC over "{sha256}.{exp}". The key is derived from the JWT secret so no new secret is needed.
"""
import hashlib
import hmac
import re
import time
from datetime import datetime, timezone

from app.core.config import get_settings

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SIG_RE = re.compile(r"[0-9a-f]{64}")
_SKEW_SECONDS = 60


def _key() -> bytes:
    return hmac.new(get_settings().JWT_SECRET_KEY.encode(), b"visuals-url-v1", hashlib.sha256).digest()


def _sign(sha256: str, exp: str) -> str:
    return hmac.new(_key(), f"{sha256}.{exp}".encode(), hashlib.sha256).hexdigest()


def sign_url(sha256: str, now: float | None = None) -> tuple[str, str]:
    """Return (url, expires_at_iso), or ("", "") when PUBLIC_BASE_URL is empty or not https."""
    s = get_settings()
    base = s.PUBLIC_BASE_URL.rstrip("/")
    if not base.startswith("https://") or len(base) == len("https://"):
        return "", ""
    exp = int((time.time() if now is None else now) + s.VISUAL_URL_TTL_SECONDS)
    sig = _sign(sha256, str(exp))
    iso = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
    return f"{base}/api/v1/visuals/files/{sha256}.webp?exp={exp}&sig={sig}", iso


def verify(sha256: str, exp: str, sig: str, now: float | None = None) -> bool:
    """Constant-time check of a signed url's parts; False (never an exception) for anything malformed or out of window."""
    try:
        if not (isinstance(sha256, str) and isinstance(exp, str) and isinstance(sig, str)):
            return False
        if not _SHA_RE.fullmatch(sha256) or not _SIG_RE.fullmatch(sig):
            return False
        if not (1 <= len(exp) <= 12 and exp.isascii() and exp.isdigit()):
            return False
        if not hmac.compare_digest(_sign(sha256, exp).encode(), sig.encode()):
            return False
        t = time.time() if now is None else now
        return t <= int(exp) <= t + get_settings().VISUAL_URL_TTL_SECONDS + _SKEW_SECONDS
    except Exception:
        return False
