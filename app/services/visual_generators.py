"""Sources of raw vehicle artwork for the visuals worker.

A generator turns (variant_key, style_version) into image bytes; the worker validates and normalizes them
afterwards, so nothing here is trusted to be a good image. The provider generator (Cloudflare Workers AI) is off
unless VISUAL_PROVIDER_ENABLED is set and both Cloudflare settings are present. It works in two steps: flux-1-schnell
draws the right car from text, then flux-2-klein-4b redraws that exact car in the style of the reference artwork.
"""
import asyncio
import base64
import binascii
import io
import re
from functools import cache
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings
from app.services.vehicle_catalog import label
from app.services.visual_image import MAX_SOURCE_BYTES

PROVIDER_TIMEOUT_SECONDS = 120
# The house style matches the reference artwork (assets/vehicles/toyota-echo.webp): pen-and-ink with a light wash,
# pale cream paint, dark olive tyres, car facing left. flux cannot make transparency and does not strip a textured
# background, so it is asked for a plain pure white one, which visual_image.normalize removes.
PROMPT_TEMPLATE = (
    "Vintage workshop manual plate: a {label} drawn in fine pen-and-ink with delicate hatching and a subtle "
    "watercolor wash, pale cream body paint, dark olive-grey tyres with hatched sidewalls, soft earthy colours, "
    "perfectly flat side elevation view with the front of the car pointing left, whole car visible, wheels on the "
    "ground, isolated on a plain solid pure white background, no texture, no paper, no shadow, no text, no people."
)
MODEL = "@cf/black-forest-labs/flux-1-schnell"
RESTYLE_MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
STYLE_REFERENCE = Path(__file__).resolve().parent.parent / "assets" / "style_reference.webp"  # the Echo artwork
RESTYLE_PROMPT = (
    "Image 1 is a car. Image 2 is a style reference. Redraw the exact car from image 1, keeping its body shape, "
    "proportions, windows, lights and wheels unchanged, but in the illustration style of image 2: the same fine "
    "pen-and-ink line weight, fine hatching, flat muted colours, pale cream body paint and dark olive-grey tyres. "
    "Strict flat side view facing left on a plain white background, even if image 1 shows the car at an angle or "
    "with a background. Do not copy the car, its shape or its doors from image 2: only its drawing style."
)
BRAVE_IMAGES_URL = "https://api.search.brave.com/res/v1/images/search"
MAX_PHOTO_BYTES = 5 * 1024 * 1024
_KEY = re.compile(r"[A-Za-z0-9_-]+")
_EXTENSIONS = (".webp", ".png", ".jpg")


class GeneratorError(Exception):
    """The generator could not produce an image; `.code` is a stable, loggable reason."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class Generator(Protocol):
    charged: bool  # True when a call can cost money; a generator may offer charged_for(variant_key) instead

    async def generate(self, variant_key: str, style_version: str) -> bytes: ...


class CuratedGenerator:
    """Reviewed images from a private directory: {directory}/{variant_key}.webp|png|jpg."""

    charged = False

    def __init__(self, directory: str):
        self._directory = Path(directory)

    def _read(self, variant_key: str) -> bytes:
        for ext in _EXTENSIONS:
            path = self._directory / f"{variant_key}{ext}"
            if not path.is_file() or path.is_symlink():
                continue
            if path.stat().st_size > MAX_SOURCE_BYTES:
                raise GeneratorError("too_large")
            return path.read_bytes()
        raise GeneratorError("no_source")

    def has_source(self, variant_key: str) -> bool:
        if not isinstance(variant_key, str) or not _KEY.fullmatch(variant_key):
            return False
        return any((self._directory / f"{variant_key}{ext}").is_file() for ext in _EXTENSIONS)

    async def generate(self, variant_key: str, style_version: str) -> bytes:
        if not isinstance(variant_key, str) or not _KEY.fullmatch(variant_key):
            raise GeneratorError("invalid_variant")
        return await asyncio.to_thread(self._read, variant_key)


_NOT_PARTS = "-mirror -headlight -parts -toy -model"
_PART_WORDS = {"mirror", "glass", "headlight", "headlamp", "lamp", "wiper", "panel", "armrest", "bumper", "grille",
               "seat", "cover", "kit", "toy", "diecast", "wheel", "tyre", "tire", "decal", "sticker"}


def _photo_candidates(results: list, name: str) -> list[str]:
    """Brave-hosted thumbnails worth trying, best first: wide photos whose title names the car and is no spare part."""
    words = re.findall(r"[a-z0-9]+", name.lower())
    scored = []
    for hit in results:
        if not isinstance(hit, dict):
            continue
        src = (hit.get("thumbnail") or {}).get("src") or ""
        host = urlparse(src).hostname or ""
        if urlparse(src).scheme != "https" or not (host == "brave.com" or host.endswith(".brave.com")):
            continue
        title = set(re.findall(r"[a-z0-9]+", str(hit.get("title", "")).lower()))
        size = hit.get("properties") or {}
        wide = (size.get("width") or 0) >= 1.3 * (size.get("height") or 1)  # product shots are square
        if wide and not title & _PART_WORDS and len(words) > 1 and words[1] in title:  # words[1] is the model, right after the make
            scored.append((sum(w in title for w in words) + ("side" in title or "profile" in title), src))
    return [src for _, src in sorted(scored, key=lambda t: -t[0])]


async def reference_photo(label: str, transport: httpx.AsyncBaseTransport | None = None) -> bytes | None:
    """A JPEG photo of the exact car from Brave Image Search, used privately as the shape reference; None if none.

    Only Brave's own thumbnail host is fetched (never an arbitrary URL from a result), and the result is re-encoded.
    """
    key = get_settings().BRAVE_API_KEY
    if not key:
        return None
    name = re.sub(r"\(.*?\)", "", label).strip()
    try:
        async with httpx.AsyncClient(transport=transport, timeout=20) as client:
            resp = await client.get(
                BRAVE_IMAGES_URL, params={"q": f"{name} car side view {_NOT_PARTS}", "count": 20},
                headers={"X-Subscription-Token": key, "Accept": "application/json"},
            )
            resp.raise_for_status()
            for hit in _photo_candidates(resp.json().get("results") or [], name):
                photo = await client.get(hit)
                if photo.status_code == 200 and len(photo.content) <= MAX_PHOTO_BYTES:
                    jpeg = await asyncio.to_thread(_as_jpeg, photo.content)
                    if jpeg:
                        return jpeg
    except (httpx.HTTPError, ValueError):
        pass
    return None


def _as_jpeg(data: bytes) -> bytes | None:
    from PIL import Image

    try:
        out = io.BytesIO()
        Image.open(io.BytesIO(data)).convert("RGB").save(out, "JPEG", quality=90)
        return out.getvalue()
    except (OSError, ValueError):
        return None


@cache
def _style_reference_png() -> bytes:
    """The reference artwork on a white background as PNG (the model wants an opaque image)."""
    from PIL import Image

    try:
        art = Image.open(STYLE_REFERENCE).convert("RGBA")
    except (OSError, ValueError) as exc:
        raise GeneratorError("style_reference_missing") from exc
    flat = Image.new("RGBA", art.size, (255, 255, 255, 255))
    flat.alpha_composite(art)
    out = io.BytesIO()
    flat.convert("RGB").save(out, "PNG")
    return out.getvalue()


class ProviderGenerator:
    """Cloudflare Workers AI, two calls per image (both use the neuron allowance, so one counted paid attempt)."""

    charged = True

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self._transport = transport

    async def _call(self, prompt: str) -> bytes:
        """Step 1: the car, drawn from text."""
        return await self._post(MODEL, json={"prompt": prompt, "steps": 4})

    async def _restyle(self, car: bytes) -> bytes:
        """Step 2: the same car in the reference artwork's style (multipart: image 1 is the car, image 2 the style)."""
        return await self._post(
            RESTYLE_MODEL,
            data={"prompt": RESTYLE_PROMPT, "width": "1024", "height": "512"},
            files={"input_image_0": ("car.jpg", car, "image/jpeg"),
                   "input_image_1": ("style.png", _style_reference_png(), "image/png")},
        )

    async def _post(self, model: str, **request) -> bytes:
        settings = get_settings()
        if not (settings.CLOUDFLARE_ACCOUNT_ID and settings.CLOUDFLARE_API_TOKEN):
            raise GeneratorError("provider_not_configured")
        url = f"https://api.cloudflare.com/client/v4/accounts/{settings.CLOUDFLARE_ACCOUNT_ID}/ai/run/{model}"
        headers = {"Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}"}
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=PROVIDER_TIMEOUT_SECONDS) as client:
                resp = await client.post(url, headers=headers, **request)
        except httpx.HTTPError as exc:
            raise GeneratorError("provider_network") from exc
        if resp.status_code != 200:
            raise GeneratorError(f"provider_http_{resp.status_code}")
        try:
            body = resp.json()
            image = (body.get("result") or body)["image"]
            return base64.b64decode(image, validate=True)
        except (ValueError, KeyError, TypeError, AttributeError, binascii.Error) as exc:
            raise GeneratorError("provider_bad_response") from exc

    async def generate(self, variant_key: str, style_version: str) -> bytes:
        name = label(variant_key)  # catalog label only, never user data
        try:
            async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS):
                # A real photo of the exact car fixes its shape; without one the text prompt has to guess it.
                car = await reference_photo(name) or await self._call(PROMPT_TEMPLATE.format(label=name))
                return await self._restyle(car)
        except TimeoutError as exc:
            raise GeneratorError("provider_timeout") from exc


class CuratedThenProvider:
    """A reviewed image from the curated directory wins; only a variant with none is generated."""

    charged = True  # the provider's; see charged_for

    def __init__(self, curated: CuratedGenerator, provider: ProviderGenerator):
        self._curated, self._provider = curated, provider

    def charged_for(self, variant_key: str) -> bool:
        return not self._curated.has_source(variant_key)

    async def generate(self, variant_key: str, style_version: str) -> bytes:
        try:
            return await self._curated.generate(variant_key, style_version)
        except GeneratorError as exc:
            if exc.code != "no_source":
                raise
        return await self._provider.generate(variant_key, style_version)


def build_generator() -> Generator:
    settings = get_settings()
    curated = CuratedGenerator(settings.VISUAL_CURATED_DIR)
    if settings.VISUAL_PROVIDER_ENABLED:
        return CuratedThenProvider(curated, ProviderGenerator())
    return curated
