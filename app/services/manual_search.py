"""Find and safely download a car's manual PDFs from the open web, using the Brave Search API.

Search results are untrusted: every URL is checked to resolve to a public address before it is fetched (also after each
redirect), the body must be a PDF within the size cap, and its first pages must mention the car's model.
"""
import asyncio
import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import urlparse, urljoin

import httpx

from app.core.config import get_settings
from app.utils.logger import get_logger_instance

logger = get_logger_instance("manual_search")

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_PAUSE_SECONDS = 1.1  # the free plan allows one request a second
MAX_REDIRECTS = 3
DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
RELEVANCE_PAGES = 8
QUERIES = (
    "{year} {make} {model} owner's manual filetype:pdf",
    "{year} {make} {model} service repair manual filetype:pdf",
)
_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _ALNUM.sub("", text.lower())


def _score(hit: dict, make: str, model: str, year: int) -> int:
    url = hit.get("url", "")
    text = _norm(f"{url} {hit.get('title', '')}")
    return (
        3 * urlparse(url).path.lower().endswith(".pdf")
        + 2 * (_norm(model) in text)
        + (_norm(make) in text)
        + (str(year) in text)
        + ("manual" in text)
    )


async def search_manual_urls(make: str, model: str, year: int, limit: int, transport: httpx.AsyncBaseTransport | None = None) -> list[str]:
    """Candidate manual URLs, best first. [] without a Brave key or on any search failure."""
    key = get_settings().BRAVE_API_KEY
    if not key or limit <= 0:
        return []
    hits: dict[str, dict] = {}
    try:
        async with httpx.AsyncClient(transport=transport, timeout=15) as client:
            for i, template in enumerate(QUERIES):
                if i:
                    await asyncio.sleep(BRAVE_PAUSE_SECONDS)
                resp = await client.get(
                    BRAVE_URL,
                    params={"q": template.format(make=make, model=model, year=year), "count": 10},
                    headers={"X-Subscription-Token": key, "Accept": "application/json"},
                )
                resp.raise_for_status()
                for hit in (resp.json().get("web") or {}).get("results") or []:
                    if isinstance(hit, dict) and isinstance(hit.get("url"), str):
                        hits.setdefault(hit["url"], hit)
    except (httpx.HTTPError, ValueError) as exc:
        await logger.warning(f"brave search failed: {type(exc).__name__}")
        if not hits:
            return []
    about_model = [h for h in hits.values() if _norm(model) in _norm(f"{h['url']} {h.get('title', '')}")]
    about_model.sort(key=lambda h: _score(h, make, model, year), reverse=True)
    return [h["url"] for h in about_model][:limit]


async def _is_public(url: str) -> bool:
    """http(s) on a standard port, whose host resolves only to public addresses.

    ponytail: resolved here and again by the HTTP client, so a DNS-rebinding host could slip through;
    pin the resolved address in a custom transport if untrusted hosts become a real threat.
    """
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.port not in (None, 80, 443):
        return False
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parts.hostname, parts.port or 443, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


async def download_manual(url: str, dest: Path, model: str, transport: httpx.AsyncBaseTransport | None = None) -> bool:
    """Download `url` to `dest` if it is a public, size-capped PDF about `model`. False when it is not."""
    cap = get_settings().MANUAL_MAX_MB * 1024 * 1024
    async with httpx.AsyncClient(transport=transport, timeout=DOWNLOAD_TIMEOUT, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            if not await _is_public(url):
                return False
            async with client.stream("GET", url, headers={"User-Agent": "Mozilla/5.0 AutoCare"}) as resp:
                if resp.is_redirect:
                    url = urljoin(url, resp.headers.get("location", ""))
                    continue
                if resp.status_code != 200 or int(resp.headers.get("content-length") or 0) > cap:
                    return False
                size = 0
                with dest.open("wb") as out:
                    async for chunk in resp.aiter_bytes():
                        size += len(chunk)
                        if size > cap:
                            return False
                        out.write(chunk)
            break
        else:
            return False
    return await asyncio.to_thread(_looks_right, dest, model)


def _looks_right(path: Path, model: str) -> bool:
    """A real PDF whose first pages name the model (so a Camry manual is not stored for a Corolla)."""
    from pypdf import PdfReader

    with path.open("rb") as f:
        if f.read(5) != b"%PDF-":
            return False
    try:
        pages = PdfReader(path).pages[:RELEVANCE_PAGES]
        text = _norm(" ".join(p.extract_text() or "" for p in pages))
    except Exception:  # pypdf raises many kinds of errors on damaged files
        return False
    return _norm(model) in text
