import hashlib
import io
import os
import struct
import warnings
import zlib

import pytest
from PIL import Image

from app.services import visual_image as vi
from app.services.visual_image import InvalidArtwork, normalize, validate_published


def _encode(img, fmt="PNG", **kw):
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def _noisy_rgba(w, h, box):
    """Transparent image with a noisy opaque rectangle (hard to compress)."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bw, bh = box[2] - box[0], box[3] - box[1]
    noise = Image.frombytes("RGB", (bw, bh), os.urandom(bw * bh * 3)).convert("RGBA")
    img.paste(noise, box[:2])
    return img


def _canvas(box):
    img = Image.new("RGBA", vi.CANVAS, (0, 0, 0, 0))
    img.paste((200, 30, 30, 255), box)
    return _encode(img, "WEBP", lossless=True)


def _alpha_bbox(data):
    return Image.open(io.BytesIO(data)).convert("RGBA").getchannel("A").getbbox()


def test_constants():
    assert vi.CANVAS == (1024, 420)
    assert vi.BASELINE_Y == 366
    assert vi.MAX_BYTES == 409_600


def test_normalize_png_to_canvas():
    src = Image.new("RGBA", (2000, 900), (0, 0, 0, 0))
    src.paste((20, 120, 220, 255), (100, 50, 1900, 850))
    out, sha, w, h = normalize(_encode(src))
    assert (w, h) == (1024, 420)
    assert sha == hashlib.sha256(out).hexdigest()
    assert len(out) <= vi.MAX_BYTES
    img = Image.open(io.BytesIO(out))
    assert img.format == "WEBP" and img.size == (1024, 420)
    assert not ({"exif", "icc_profile", "xmp"} & set(img.info))
    box = _alpha_bbox(out)
    assert box[3] - 1 == 366
    assert box[0] >= 16 and 1024 - box[2] >= 16 and box[1] >= 8
    validate_published(out)


def test_normalize_strips_metadata():
    src = Image.new("RGB", (300, 200), (255, 255, 255))
    src.paste((10, 200, 10), (50, 50, 250, 150))
    exif = Image.Exif()
    exif[0x010F] = "SecretCam"
    data = _encode(src, "JPEG", exif=exif.tobytes(), icc_profile=b"x" * 64)
    out = normalize(data)[0]
    assert b"SecretCam" not in out
    assert not ({"exif", "icc_profile", "xmp"} & set(Image.open(io.BytesIO(out)).info))


def test_opaque_source_background_removed():
    src = Image.new("RGB", (800, 400), (255, 255, 255))
    src.paste((200, 20, 20), (200, 100, 600, 300))
    out = normalize(_encode(src))[0]
    img = Image.open(io.BytesIO(out)).convert("RGBA")
    assert img.getpixel((2, 2))[3] == 0
    assert img.getchannel("A").getbbox() is not None
    validate_published(out)


def test_uniform_image_is_bad_margins():
    with pytest.raises(InvalidArtwork) as e:
        normalize(_encode(Image.new("RGB", (100, 100), (5, 5, 5))))
    assert e.value.code == "bad_margins"
    with pytest.raises(InvalidArtwork) as e:
        normalize(_encode(Image.new("RGBA", (100, 100), (0, 0, 0, 0))))
    assert e.value.code == "bad_margins"


def test_first_frame_only():
    a = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    a.paste((255, 0, 0, 255), (20, 20, 180, 80))
    b = Image.new("RGBA", (200, 100), (0, 255, 0, 255))
    data = _encode(a, "WEBP", save_all=True, append_images=[b], lossless=True)
    out = normalize(data)[0]
    px = Image.open(io.BytesIO(out)).convert("RGBA").getpixel((512, 300))
    assert px[:3] == (255, 0, 0)


def test_lossy_fallback_keeps_alpha_and_budget():
    src = _noisy_rgba(1200, 600, (0, 0, 1200, 600))
    out = normalize(_encode(src))[0]
    assert len(out) <= vi.MAX_BYTES
    box = _alpha_bbox(out)
    assert box[3] - 1 == 366
    validate_published(out)


def test_too_many_bytes(monkeypatch):
    monkeypatch.setattr(vi, "MAX_BYTES", 100)
    src = _noisy_rgba(600, 300, (0, 0, 600, 300))
    with pytest.raises(InvalidArtwork) as e:
        normalize(_encode(src))
    assert e.value.code == "too_many_bytes"


def test_over_pixel_cap_rejected_before_decode(monkeypatch):
    data = _encode(Image.new("RGB", (5000, 5000), (1, 2, 3)))
    monkeypatch.setattr(vi, "MAX_SOURCE_PIXELS", 1_000_000)

    def boom(self, *a, **k):
        raise AssertionError("decoded before the size check")

    monkeypatch.setattr(Image.Image, "load", boom)
    monkeypatch.setattr(Image.Image, "convert", boom)
    with pytest.raises(InvalidArtwork) as e:
        normalize(data)
    assert e.value.code == "too_large"


def test_source_byte_cap(monkeypatch):
    monkeypatch.setattr(vi, "MAX_SOURCE_BYTES", 10)
    with pytest.raises(InvalidArtwork) as e:
        normalize(b"x" * 11)
    assert e.value.code == "too_large"


def test_decompression_bomb_rejected():
    # tiny PNG declaring 40000x40000 (1.6e9 px): Pillow raises DecompressionBombError at open
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d))

    ihdr = struct.pack(">IIBBBBB", 40000, 40000, 8, 0, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\0")) + chunk(b"IEND", b"")
    before = Image.MAX_IMAGE_PIXELS
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(InvalidArtwork) as e:
            normalize(data)
    assert e.value.code == "too_large"
    assert Image.MAX_IMAGE_PIXELS == before


def test_warning_level_bomb_rejected(monkeypatch):
    # 10k px against a 1000 px global cap is inside the warning zone only if below 2x; use 1500 so it warns
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 6000)
    data = _encode(Image.new("RGB", (100, 100), (1, 2, 3)))  # 10k px: > cap (warning), < 2x cap
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(InvalidArtwork) as e:
            normalize(data)
    assert e.value.code == "too_large"


@pytest.mark.parametrize("data", [os.urandom(500), b"", b"GIF89a" + b"\0" * 20])
def test_not_image(data):
    with pytest.raises(InvalidArtwork) as e:
        normalize(data)
    assert e.value.code == "not_image"


def test_truncated_png_is_not_image():
    data = _encode(Image.new("RGB", (400, 400), (9, 9, 9)))
    with pytest.raises(InvalidArtwork) as e:
        normalize(data[: len(data) // 2])
    assert e.value.code == "not_image"


def test_disallowed_format_is_not_image():
    with pytest.raises(InvalidArtwork) as e:
        normalize(_encode(Image.new("RGB", (50, 50), (9, 9, 9)), "GIF"))
    assert e.value.code == "not_image"


def test_validate_published_ok():
    data = _canvas((100, 100, 900, 367))
    sha, w, h = validate_published(data)
    assert (sha, w, h) == (hashlib.sha256(data).hexdigest(), 1024, 420)


def test_validate_wrong_size():
    data = _encode(Image.new("RGBA", (512, 210), (0, 0, 0, 0)), "WEBP", lossless=True)
    with pytest.raises(InvalidArtwork) as e:
        validate_published(data)
    assert e.value.code == "bad_dimensions"


@pytest.mark.parametrize(
    "box",
    [
        (0, 100, 900, 367),  # touches left edge
        (100, 100, 1024, 367),  # touches right edge
        (100, 0, 900, 367),  # touches top
        (100, 100, 900, 400),  # baseline too low
        (100, 100, 900, 300),  # baseline too high
    ],
)
def test_validate_bad_margins(box):
    with pytest.raises(InvalidArtwork) as e:
        validate_published(_canvas(box))
    assert e.value.code == "bad_margins"


def test_validate_not_webp_or_garbage():
    for data in (os.urandom(100), _encode(Image.new("RGBA", vi.CANVAS, (0, 0, 0, 0)), "PNG")):
        with pytest.raises(InvalidArtwork) as e:
            validate_published(data)
        assert e.value.code == "not_image"


def test_validate_too_many_bytes(monkeypatch):
    data = _canvas((100, 100, 900, 367))
    monkeypatch.setattr(vi, "MAX_BYTES", len(data) - 1)
    with pytest.raises(InvalidArtwork) as e:
        validate_published(data)
    assert e.value.code == "too_many_bytes"
