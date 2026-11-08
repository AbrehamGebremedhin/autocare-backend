"""Sources of raw vehicle artwork for the visuals worker.

A generator turns (variant_key, style_version) into image bytes; the worker validates and normalizes them
afterwards, so nothing here is trusted to be a good image. The provider generator is off unless
VISUAL_PROVIDER_ENABLED is set, and its network call is deliberately left unimplemented.
"""
import asyncio
import re
from pathlib import Path
from typing import Protocol

from app.core.config import get_settings
from app.services.vehicle_catalog import label
from app.services.visual_image import MAX_SOURCE_BYTES

PROVIDER_TIMEOUT_SECONDS = 120
PROMPT_TEMPLATE = "A clean side-view illustration of a {label}, flat workshop style, on a transparent background."
_KEY = re.compile(r"[A-Za-z0-9_-]+")
_EXTENSIONS = (".webp", ".png", ".jpg")


class GeneratorError(Exception):
    """The generator could not produce an image; `.code` is a stable, loggable reason."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class Generator(Protocol):
    charged: bool  # True when a call can cost money

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

    async def generate(self, variant_key: str, style_version: str) -> bytes:
        if not isinstance(variant_key, str) or not _KEY.fullmatch(variant_key):
            raise GeneratorError("invalid_variant")
        return await asyncio.to_thread(self._read, variant_key)


class ProviderGenerator:
    """Paid image provider. Only built when VISUAL_PROVIDER_ENABLED; `_call` is the single place a real
    service would be wired in, and it is not wired to any."""

    charged = True

    async def _call(self, prompt: str) -> bytes:
        raise GeneratorError("provider_not_configured")

    async def generate(self, variant_key: str, style_version: str) -> bytes:
        prompt = PROMPT_TEMPLATE.format(label=label(variant_key))  # catalog label only, never user data
        try:
            async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS):
                return await self._call(prompt)
        except TimeoutError as exc:
            raise GeneratorError("provider_timeout") from exc


def build_generator() -> Generator:
    settings = get_settings()
    if settings.VISUAL_PROVIDER_ENABLED:
        return ProviderGenerator()
    return CuratedGenerator(settings.VISUAL_CURATED_DIR)
