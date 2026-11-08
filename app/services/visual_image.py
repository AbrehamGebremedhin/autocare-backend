"""Normalization and validation of vehicle artwork.

Pure functions on bytes. The published artwork is a 1024x420 WebP with the ground baseline at y=366, no
metadata and at most MAX_BYTES. `normalize` produces it from an untrusted source image; `validate_published`
checks (without transforming) that bytes meet that contract.
"""
import hashlib
import io
import warnings

from PIL import Image, ImageChops

CANVAS = (1024, 420)
BASELINE_Y = 366
MAX_BYTES = 409_600
MAX_SOURCE_PIXELS = 25_000_000
MAX_SOURCE_BYTES = 20_000_000

# a pixel counts as artwork only at or above this alpha; fainter pixels (stray specks, soft shadows) are ignored
ALPHA_MIN = 128
_SIDE_MARGIN = 16
_TOP_MARGIN = 8
_BG_TOLERANCE = 12
_LOSSY_QUALITIES = (90, 80, 70, 60, 50)
_SOURCE_FORMATS = {"PNG", "JPEG", "WEBP"}
_METADATA_KEYS = ("exif", "icc_profile", "xmp")


class InvalidArtwork(Exception):
    """Artwork rejected; `.code` is one of too_large, not_image, bad_dimensions, bad_margins, too_many_bytes."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _content_bbox(rgba: Image.Image):
    """Bounding box of the pixels with alpha >= ALPHA_MIN, or None."""
    return rgba.getchannel("A").point(lambda v: 255 if v >= ALPHA_MIN else 0).getbbox()


def _open(data: bytes) -> Image.Image:
    """Open lazily (headers only) without leaving a bomb warning or global state behind."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            return Image.open(io.BytesIO(data))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidArtwork("too_large") from exc
    except (Image.UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError) as exc:
        raise InvalidArtwork("not_image") from exc


def _load_rgba(img: Image.Image) -> Image.Image:
    try:
        img.seek(0)
        return img.convert("RGBA")
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError) as exc:
        raise InvalidArtwork("not_image") from exc


def _strip_flat_background(rgba: Image.Image) -> Image.Image:
    """For fully opaque sources, make pixels close to the top-left colour transparent."""
    if rgba.getchannel("A").getextrema()[0] != 255:
        return rgba
    bg = Image.new("RGB", rgba.size, rgba.getpixel((0, 0))[:3])
    r, g, b = ImageChops.difference(rgba.convert("RGB"), bg).split()
    diff = ImageChops.lighter(ImageChops.lighter(r, g), b)
    rgba = rgba.copy()
    rgba.putalpha(diff.point(lambda v: 255 if v > _BG_TOLERANCE else 0))
    return rgba


def _compose(rgba: Image.Image) -> Image.Image:
    bbox = _content_bbox(rgba)
    if bbox is None:
        raise InvalidArtwork("bad_margins")
    art = rgba.crop(bbox)
    max_w = CANVAS[0] - 2 * _SIDE_MARGIN
    max_h = BASELINE_Y + 1 - _TOP_MARGIN
    scale = min(max_w / art.width, max_h / art.height)
    size = (max(1, min(max_w, round(art.width * scale))), max(1, min(max_h, round(art.height * scale))))
    art = art.resize(size, Image.LANCZOS)
    # resampling can fade the outermost rows to alpha 0; re-trim so the lowest visible row is the baseline
    trim = _content_bbox(art)
    if trim is None:
        raise InvalidArtwork("bad_margins")
    art = art.crop(trim)
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    canvas.paste(art, ((CANVAS[0] - art.width) // 2, BASELINE_Y + 1 - art.height))
    return canvas


def _encode(canvas: Image.Image) -> bytes:
    """Lossless first, then lossy quality steps; metadata is never written."""
    base = {"exif": b"", "xmp": b"", "icc_profile": None}
    attempts = [{"lossless": True, "quality": 100, "method": 4}] + [
        {"lossless": False, "quality": q, "alpha_quality": 100, "method": 4} for q in _LOSSY_QUALITIES
    ]
    for opts in attempts:
        buf = io.BytesIO()
        canvas.save(buf, "WEBP", **base, **opts)
        out = buf.getvalue()
        if len(out) <= MAX_BYTES:
            return out
    raise InvalidArtwork("too_many_bytes")


def normalize(data: bytes) -> tuple[bytes, str, int, int]:
    """Turn untrusted image bytes into the published artwork: (webp_bytes, sha256_hex, width, height)."""
    if len(data) > MAX_SOURCE_BYTES:
        raise InvalidArtwork("too_large")
    img = _open(data)
    if img.format not in _SOURCE_FORMATS:
        raise InvalidArtwork("not_image")
    if img.width * img.height > MAX_SOURCE_PIXELS:
        raise InvalidArtwork("too_large")
    canvas = _compose(_strip_flat_background(_load_rgba(img)))
    out = _encode(canvas)
    validate_published(out)
    return out, hashlib.sha256(out).hexdigest(), CANVAS[0], CANVAS[1]


def validate_published(data: bytes) -> tuple[str, int, int]:
    """Check bytes already meet the published contract; returns (sha256_hex, width, height)."""
    img = _open(data)
    if img.format != "WEBP":
        raise InvalidArtwork("not_image")
    if img.size != CANVAS:
        raise InvalidArtwork("bad_dimensions")
    if len(data) > MAX_BYTES:
        raise InvalidArtwork("too_many_bytes")
    if getattr(img, "is_animated", False) or getattr(img, "n_frames", 1) != 1:
        raise InvalidArtwork("not_image")
    if any(img.info.get(k) for k in _METADATA_KEYS):
        raise InvalidArtwork("not_image")
    bbox = _content_bbox(_load_rgba(img))
    if bbox is None:
        raise InvalidArtwork("bad_margins")
    left, top, right, bottom = bbox
    if bottom - 1 != BASELINE_Y or left < _SIDE_MARGIN or right > CANVAS[0] - _SIDE_MARGIN or top < _TOP_MARGIN:
        raise InvalidArtwork("bad_margins")
    return hashlib.sha256(data).hexdigest(), CANVAS[0], CANVAS[1]
