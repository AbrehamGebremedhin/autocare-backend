import asyncio
import io
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

from app.core.config import get_settings
from app.services import media
from app.services.media import (
    MediaError,
    audio_pcm,
    extract_frames,
    probe,
    validate_uploads,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _ff(*args):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    tv = ["-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10"]
    sine = ["-f", "lavfi", "-i", "sine=frequency=440:duration=3"]
    _ff(*tv, *sine, "-shortest", "-pix_fmt", "yuv420p", str(d / "clip.mp4"))
    _ff(*tv, "-pix_fmt", "yuv420p", str(d / "silent.mp4"))
    _ff(*tv, "-pix_fmt", "yuv420p", str(d / "silent2.mp4"))
    for side in (7680, 7682):  # MAX_VIDEO_SIDE (8K) and just over it
        _ff("-f", "lavfi", "-i", f"testsrc=duration=1:size={side}x16:rate=1", "-pix_fmt", "yuv420p", str(d / f"wide{side}.mp4"))
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=61", "-ar", "8000", str(d / "long.wav"))
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-ar", "8000", str(d / "tone.wav"))
    _ff("-f", "lavfi", "-i", "testsrc=duration=90:size=64x48:rate=1", "-pix_fmt", "yuv420p", str(d / "ninetyvid.mp4"))
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=150", "-ar", "8000", str(d / "toolong.wav"))
    buf = io.BytesIO()
    Image.new("RGB", (80, 60), (200, 0, 0)).save(
        buf, "MPO", save_all=True, append_images=[Image.new("RGB", (80, 60), (0, 0, 200))]
    )
    (d / "multi.mpo").write_bytes(buf.getvalue())
    _ff(
        "-f", "lavfi", "-i", "testsrc=duration=121:size=64x48:rate=1",
        "-pix_fmt", "yuv420p", str(d / "longvid.mp4"),
    )
    (d / "fake.mp4").write_text("this is not a video, just text\n")
    Image.new("RGBA", (2000, 1000), (255, 0, 0, 128)).save(d / "alpha.png")
    Image.new("RGB", (64, 64), (0, 255, 0)).save(d / "small.webp")
    Image.new("RGB", (64, 64), (0, 0, 255)).save(d / "small.gif")
    Image.new("RGB", (64, 64), (0, 0, 255)).save(d / "small.jpg")
    Image.new("RGB", (64, 64), (0, 0, 255)).save(d / "small.bmp")
    # cover-art-only audio: mp3 with an attached picture
    Image.new("RGB", (64, 64), (9, 9, 9)).save(d / "cover.jpg")
    _ff(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-i", str(d / "cover.jpg"),
        "-map", "0", "-map", "1", "-c:a", "libmp3lame", "-c:v", "copy",
        "-disposition:v", "attached_pic", str(d / "art.mp3"),
    )
    return d


@pytest.fixture
def tmp_media(media_dir, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    opened = []

    def upload(name, filename=None):
        f = open(media_dir / name, "rb")
        opened.append(f)
        return UploadFile(file=f, filename=filename or name)

    yield SimpleNamespace(dir=media_dir, work=work, upload=upload)
    for f in opened:
        f.close()


async def test_video_with_audio_classified(tmp_media):
    [u] = await validate_uploads([tmp_media.upload("clip.mp4")], tmp_media.work)
    assert u.kind == "video" and u.has_audio and 2.5 < u.duration < 3.5


async def test_silent_video_ok(tmp_media):
    [u] = await validate_uploads([tmp_media.upload("silent.mp4")], tmp_media.work)
    assert u.kind == "video" and not u.has_audio


async def test_fake_mp4_rejected(tmp_media):
    with pytest.raises(HTTPException) as e:
        await validate_uploads([tmp_media.upload("fake.mp4")], tmp_media.work)
    assert e.value.status_code == 415


async def test_long_audio_rejected(tmp_media):
    with pytest.raises(HTTPException) as e:
        await validate_uploads([tmp_media.upload("long.wav")], tmp_media.work)
    assert e.value.status_code == 413 and "Audio clips must be 60 seconds" in e.value.detail


async def test_long_video_rejected(tmp_media):
    with pytest.raises(HTTPException) as e:
        await validate_uploads([tmp_media.upload("longvid.mp4")], tmp_media.work)
    assert e.value.status_code == 413 and "Video clips must be 120 seconds" in e.value.detail


async def test_video_up_to_120_seconds_is_accepted(tmp_media):
    [u] = await validate_uploads([tmp_media.upload("ninetyvid.mp4")], tmp_media.work)  # over the 60 s audio limit
    assert u.kind == "video" and 89 < u.duration < 91


async def test_frames_evenly_spaced_and_jpeg(tmp_media):
    frames = await extract_frames(tmp_media.dir / "clip.mp4", 3.0, 8)
    assert len(frames) == 8 and all(f[:2] == b"\xff\xd8" for f in frames)


async def test_frames_fewer_for_sub_second_clip(tmp_media):
    frames = await extract_frames(tmp_media.dir / "clip.mp4", 0.25, 8)
    assert 1 <= len(frames) < 8


async def test_audio_pcm_shape(tmp_media):
    pcm = await audio_pcm(tmp_media.dir / "clip.mp4")
    assert pcm.dtype == np.float32 and abs(len(pcm) / 48000 - 3.0) < 0.1


async def test_audio_only_classified_audio(tmp_media):
    [u] = await validate_uploads([tmp_media.upload("tone.wav")], tmp_media.work)
    assert u.kind == "audio" and u.has_audio and 1.5 < u.duration < 2.5


async def test_cover_art_is_not_video(tmp_media):
    [u] = await validate_uploads([tmp_media.upload("art.mp3")], tmp_media.work)
    assert u.kind == "audio"


async def test_probe_reports_streams(tmp_media):
    info = await probe(tmp_media.dir / "silent.mp4")
    assert info["has_video"] and not info["has_audio"] and 2.5 < info["duration"] < 3.5
    assert info["max_side"] == 320


async def test_video_resolution_capped(tmp_media):
    [u] = await validate_uploads([tmp_media.upload("wide7680.mp4")], tmp_media.work)
    assert u.kind == "video"
    with pytest.raises(HTTPException) as e:
        await validate_uploads([tmp_media.upload("wide7682.mp4")], tmp_media.work)
    assert e.value.status_code == 413 and "7680" in e.value.detail


async def test_probe_and_pcm_raise_media_error_on_corrupt(tmp_media):
    with pytest.raises(MediaError):
        await probe(tmp_media.dir / "fake.mp4")
    with pytest.raises(MediaError):
        await audio_pcm(tmp_media.dir / "fake.mp4")


async def test_png_and_webp_reencoded_to_small_jpeg(tmp_media):
    out = await validate_uploads(
        [tmp_media.upload("alpha.png"), tmp_media.upload("small.webp")], tmp_media.work
    )
    assert len(out) == 2
    for u in out:
        assert u.kind == "image" and u.path.suffix == ".jpg"
        assert u.duration is None and u.has_audio is False
        assert u.path.read_bytes()[:2] == b"\xff\xd8"
        with Image.open(u.path) as im:
            assert im.format == "JPEG" and max(im.size) <= 1024
    with Image.open(out[0].path) as im:
        assert im.size == (1024, 512)


async def test_gif_and_bmp_rejected(tmp_media):
    for name in ("small.gif", "small.bmp"):
        with pytest.raises(HTTPException) as e:
            await validate_uploads([tmp_media.upload(name)], tmp_media.work)
        assert e.value.status_code == 415


async def test_oversized_image_413(tmp_media, monkeypatch):
    monkeypatch.setattr(get_settings(), "MAX_IMAGE_MB", 0)
    with pytest.raises(HTTPException) as e:
        await validate_uploads([tmp_media.upload("small.jpg")], tmp_media.work)
    assert e.value.status_code == 413


async def test_decompression_bomb_rejected(tmp_media, monkeypatch):
    monkeypatch.setattr(media, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(HTTPException) as e:
        await validate_uploads([tmp_media.upload("small.jpg")], tmp_media.work)
    assert e.value.status_code == 413


async def test_too_many_images_422(tmp_media):
    files = [tmp_media.upload("small.jpg") for _ in range(5)]
    with pytest.raises(HTTPException) as e:
        await validate_uploads(files, tmp_media.work)
    assert e.value.status_code == 422


async def test_four_images_ok(tmp_media):
    files = [tmp_media.upload("small.jpg") for _ in range(4)]
    assert len(await validate_uploads(files, tmp_media.work)) == 4


async def test_image_plus_audio_422(tmp_media):
    with pytest.raises(HTTPException) as e:
        await validate_uploads(
            [tmp_media.upload("small.jpg"), tmp_media.upload("tone.wav")], tmp_media.work
        )
    assert e.value.status_code == 422


async def test_two_videos_422(tmp_media):
    with pytest.raises(HTTPException) as e:
        await validate_uploads(
            [tmp_media.upload("silent.mp4"), tmp_media.upload("silent2.mp4")], tmp_media.work
        )
    assert e.value.status_code == 422


async def test_no_files_returns_empty(tmp_media):
    assert await validate_uploads([], tmp_media.work) == []


async def test_traversal_filename_stays_in_tmpdir(tmp_media):
    out = await validate_uploads(
        [tmp_media.upload("clip.mp4", filename="../../evil.mp4")], tmp_media.work
    )
    root = tmp_media.work.resolve()
    assert all(root in u.path.resolve().parents for u in out)
    assert not (tmp_media.work.parent.parent / "evil.mp4").exists()


async def test_audio_pcm_decode_is_capped(tmp_media):
    # 150 s of audio: the decode itself must stop at the longest allowed clip (the video limit).
    pcm = await audio_pcm(tmp_media.dir / "toolong.wav")
    assert 0 < len(pcm) <= get_settings().MAX_VIDEO_SECONDS * 48000


async def test_mpo_jpeg_accepted_and_reencoded(tmp_media):
    [u] = await validate_uploads([tmp_media.upload("multi.mpo")], tmp_media.work)
    assert u.kind == "image" and u.path.read_bytes()[:2] == b"\xff\xd8"
    with Image.open(u.path) as im:
        assert im.format == "JPEG"


async def test_media_error_message_has_no_paths(tmp_media):
    target = tmp_media.dir / "fake.mp4"
    with pytest.raises(MediaError) as e:
        await probe(target)
    assert str(e.value) == "Could not process media."
    assert str(tmp_media.dir) not in str(e.value) and "fake.mp4" not in str(e.value)
    assert e.value.detail  # raw tool output kept for logs


async def test_image_cap_applies_only_to_images(tmp_media, monkeypatch):
    # Image cap of 0 MB: any image is 413, an audio file of any size is not.
    monkeypatch.setattr(get_settings(), "MAX_IMAGE_MB", 0)
    with pytest.raises(HTTPException) as e:
        await validate_uploads([tmp_media.upload("small.jpg")], tmp_media.work)
    assert e.value.status_code == 413
    [u] = await validate_uploads([tmp_media.upload("tone.wav")], tmp_media.work)
    assert u.kind == "audio"


async def test_timeout_kills_and_reaps_process(monkeypatch):
    monkeypatch.setattr(media, "SUBPROCESS_TIMEOUT", 0.5)
    procs = []
    real = asyncio.create_subprocess_exec

    async def spy(*a, **k):
        p = await real(*a, **k)
        procs.append(p)
        return p

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    with pytest.raises(MediaError):
        # endless source with no -t: would run forever without the timeout
        await media._run(
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine", "-f", "null", "-"
        )
    assert procs and procs[0].returncode is not None
