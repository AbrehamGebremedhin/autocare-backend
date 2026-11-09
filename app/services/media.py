import asyncio
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import get_settings


@dataclass
class Upload:
    kind: Literal["image", "audio", "video"]
    path: Path
    duration: float | None
    has_audio: bool


# --- upload validation and ffmpeg helpers -----------------------------------

MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb cap (width * height)
MAX_IMAGE_SIDE = 1024
MAX_VIDEO_SIDE = 4096  # px; bounds the memory of each concurrent frame decode
SUBPROCESS_TIMEOUT = 30.0
CHUNK = 1024 * 1024
_ALLOWED_IMAGE_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}  # MPO: multi-picture JPEG (phones)
_MB = 1024 * 1024


class MediaError(Exception):
    """ffmpeg/ffprobe failed (non-zero exit, timeout, unreadable output).

    str(e) is generic and safe to show clients; raw tool output (which can
    contain server paths) is kept in `.detail` for logging only.
    """

    def __init__(self, message: str = "Could not process media.", detail: str = ""):
        super().__init__(message)
        self.detail = detail


async def _run(*argv: str) -> bytes:
    """Run a binary (argument list, no shell) and return stdout."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), SUBPROCESS_TIMEOUT)
    except asyncio.TimeoutError:
        raise MediaError(detail=f"{argv[0]} timed out") from None
    finally:
        # Always reap: timeout, cancellation or any other error.
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
    if proc.returncode != 0:
        raise MediaError(detail=err.decode("utf-8", "replace").strip() or f"{argv[0]} failed")
    return out


async def probe(path: Path) -> dict:
    out = await _run(
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_streams", "-show_format", "-of", "json", "-i", str(path)
    )
    try:
        info = json.loads(out)
    except ValueError as exc:
        raise MediaError("unreadable ffprobe output") from exc
    streams = info.get("streams") or []
    video = [
        s
        for s in streams
        if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
    ]
    audio = [s for s in streams if s.get("codec_type") == "audio"]

    def _num(v) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return f if math.isfinite(f) and f > 0 else 0.0

    duration = _num((info.get("format") or {}).get("duration"))
    if not duration:
        duration = max((_num(s.get("duration")) for s in video + audio), default=0.0)
    max_side = max((_num(s.get(k)) for s in video for k in ("width", "height")), default=0.0)
    return {"duration": duration, "has_audio": bool(audio), "has_video": bool(video), "max_side": max_side}


def decode_limit() -> int:
    """Seconds of any clip that are ever decoded: the longer of the audio and video limits. Validation holds each
    kind to its own limit; this caps decoding itself, whatever a container claims."""
    s = get_settings()
    return max(s.MAX_MEDIA_SECONDS, s.MAX_VIDEO_SECONDS)


async def audio_pcm(path: Path, sr: int = 48000) -> np.ndarray:
    # The container's Duration is attacker-controlled, so cap the decode itself:
    # -t as an output option means ffmpeg never emits more than the limit.
    limit = decode_limit()
    out = await _run(
        "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sr), "-t", str(limit), "-f", "f32le", "-",
    )  # fmt: skip
    max_bytes = limit * sr * 4
    if len(out) > max_bytes + 4 * sr:  # small slack for resampler padding
        raise MediaError(detail="decoded audio exceeds limit")
    return np.frombuffer(out[: max_bytes - max_bytes % 4], dtype="<f4").astype(np.float32, copy=True)


async def extract_frames(path: Path, duration: float, n: int) -> list[bytes]:
    # "Fewer frames if the clip is short": only clips under 1 s get fewer than n
    # frames (about 8 per second); anything longer yields the full n.
    n_eff = n if duration >= 1.0 else max(1, round(duration * 8))
    n_eff = min(n, n_eff)

    async def one(i: int) -> bytes | None:
        # Clamp: a lying container must not make us seek far past the limit.
        t = min(duration * (i + 0.5) / n_eff, decode_limit() - 0.001)
        try:
            return await _run(
                "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-ss", f"{t:.3f}", "-i", str(path),
                "-frames:v", "1", "-vf", f"scale='min({MAX_IMAGE_SIDE},iw)':-2",
                "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-",
            )  # fmt: skip
        except MediaError:
            return None

    frames = await asyncio.gather(*(one(i) for i in range(n_eff)))
    return [f for f in frames if f]


def _image_to_jpeg(src: Path, dst: Path) -> str:
    """Return "ok", "not_image", "bad_format" or "too_big"; on "ok" dst holds a JPEG."""
    try:
        with Image.open(src) as im:
            if im.format not in _ALLOWED_IMAGE_FORMATS:
                return "bad_format"
            if im.width * im.height > MAX_IMAGE_PIXELS:
                return "too_big"
            im.verify()
        with Image.open(src) as im:  # verify() invalidates the handle
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            im.save(dst, "JPEG", quality=85)
    except Image.DecompressionBombError:
        return "too_big"
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return "not_image" if not dst.exists() else "bad_format"
    return "ok"


def _looks_like_image(head: bytes) -> bool:
    return head[:3] == b"\xff\xd8\xff" or head[:8] == b"\x89PNG\r\n\x1a\n" or (
        head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    )


async def _save_upload(f: UploadFile, dest: Path, budget: list[int]) -> int:
    """Stream to dest in chunks; 413 as soon as a limit is crossed. budget = [bytes left in request]."""
    s = get_settings()
    size = 0
    limit = None
    with open(dest, "wb") as out:
        while chunk := await f.read(CHUNK):
            if limit is None:
                limit = (s.MAX_IMAGE_MB if _looks_like_image(chunk) else s.MAX_MEDIA_MB) * _MB
            size += len(chunk)
            budget[0] -= len(chunk)
            if size > limit or budget[0] < 0:
                raise HTTPException(413, "File is too large.")
            out.write(chunk)
    return size


async def validate_uploads(files: list[UploadFile], tmpdir: Path) -> list[Upload]:
    s = get_settings()
    if not files:
        return []
    if len(files) > s.MAX_IMAGES:
        raise HTTPException(422, f"At most {s.MAX_IMAGES} images, or a single audio/video file.")
    budget = [max(s.MAX_MEDIA_MB, s.MAX_IMAGES * s.MAX_IMAGE_MB) * _MB]
    uploads: list[Upload] = []
    for f in files:
        # Server-generated name: the client filename is never used for a path.
        raw = tmpdir / f"{uuid.uuid4().hex}.bin"
        await _save_upload(f, raw, budget)
        jpg = tmpdir / f"{uuid.uuid4().hex}.jpg"
        status = await asyncio.to_thread(_image_to_jpeg, raw, jpg)
        if status == "too_big":
            raise HTTPException(413, "Image dimensions are too large.")
        if status == "bad_format":
            raise HTTPException(415, "Only JPEG, PNG and WebP images are supported.")
        if status == "ok":
            raw.unlink(missing_ok=True)
            uploads.append(Upload("image", jpg, None, False))
            continue
        jpg.unlink(missing_ok=True)
        try:
            info = await probe(raw)
        except MediaError:
            raise HTTPException(415, "Unsupported file type.") from None
        if info["has_video"] and info["duration"] > 0:
            kind = "video"
        elif info["has_audio"] and not info["has_video"] and info["duration"] > 0:
            kind = "audio"
        else:
            raise HTTPException(415, "Unsupported file type.")
        max_seconds = s.MAX_VIDEO_SECONDS if kind == "video" else s.MAX_MEDIA_SECONDS
        if info["duration"] > max_seconds:
            raise HTTPException(413, f"{kind.capitalize()} clips must be {max_seconds} seconds or shorter.")
        if kind == "video" and info["max_side"] > MAX_VIDEO_SIDE:
            raise HTTPException(413, f"Video resolution must be at most {MAX_VIDEO_SIDE} pixels per side.")
        uploads.append(Upload(kind, raw, info["duration"], info["has_audio"]))

    images = [u for u in uploads if u.kind == "image"]
    if images and len(images) != len(uploads):
        raise HTTPException(422, "Send images, or a single audio/video file, not both.")
    if not images and len(uploads) != 1:
        raise HTTPException(422, "Send only one audio or video file.")
    return uploads
