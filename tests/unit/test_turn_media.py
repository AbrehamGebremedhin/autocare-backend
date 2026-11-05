"""Media turns: photos, audio and video through run_turn and the send route. All offline fakes."""

import asyncio
import base64
import gc
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from app.api.v1 import session_route
from app.db.bucket_operations import SupabaseBucketManager
from app.schemas.session import SoundTag
from app.services import diagnosis, hearing, media
from app.services.diagnosis import run_turn
from app.services.media import MediaError, Upload
from app.utils.limiter import limiter

LLM_ERROR = "The diagnosis service failed, please retry."
MEDIA_ERROR = "Could not process the attached media."
TRANSCRIPT = "it squeals when I start it cold"
SQUEAL = SoundTag(label="belt squeal", score=0.70)
PCM = np.zeros(48000, dtype=np.float32)
PROCESSING = ("stage", {"name": "processing_media"})
SEARCHING = ("stage", {"name": "searching"})


def _jpeg(color=(200, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buf, "JPEG")
    return buf.getvalue()


FRAMES = [_jpeg((i * 30, 0, 0)) for i in range(8)]


def _parse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.split("\n\n"):
        if block:
            event, data = block.split("\n")
            events.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def _image_blocks(prompt: list[dict]) -> list[bytes]:
    prefix = "data:image/jpeg;base64,"
    return [
        base64.b64decode(b["image_url"]["url"].removeprefix(prefix))
        for b in prompt[-1]["content"]
        if b["type"] == "image_url"
    ]


@pytest.fixture(autouse=True)
def _clean():
    limiter.reset()
    session_route._in_flight.clear()
    session_route._deleting.clear()
    yield
    session_route._in_flight.clear()
    session_route._deleting.clear()


@pytest.fixture
def fake_hearing(monkeypatch):
    fake = SimpleNamespace(transcribe=AsyncMock(return_value=TRANSCRIPT), tag_sounds=AsyncMock(return_value=[SQUEAL]))
    monkeypatch.setattr(hearing, "transcribe", fake.transcribe)
    monkeypatch.setattr(hearing, "tag_sounds", fake.tag_sounds)
    return fake


@pytest.fixture
def fake_media(monkeypatch):
    fake = SimpleNamespace(extract_frames=AsyncMock(return_value=FRAMES), audio_pcm=AsyncMock(return_value=PCM))
    monkeypatch.setattr(media, "extract_frames", fake.extract_frames)
    monkeypatch.setattr(media, "audio_pcm", fake.audio_pcm)
    return fake


@pytest.fixture
def fake_all(fake_llm, fake_kb, fake_store, fake_bucket, fake_hearing, fake_media):
    return SimpleNamespace(
        llm=fake_llm, kb=fake_kb, store=fake_store, bucket=fake_bucket, hearing=fake_hearing, media=fake_media
    )


@pytest.fixture
def make_upload(tmp_path):
    def make(kind, data: bytes, *, duration=None, has_audio=False, folder: Path | None = None) -> Upload:
        path = (folder or tmp_path) / f"{uuid4().hex}.{'jpg' if kind == 'image' else 'bin'}"
        path.write_bytes(data)
        return Upload(kind, path, duration, has_audio)

    return make


@pytest.fixture
def video_upload(make_upload):
    return make_upload("video", b"original video bytes", duration=3.0, has_audio=True)


@pytest.fixture
def silent_video_upload(make_upload):
    return make_upload("video", b"original silent video", duration=3.0, has_audio=False)


@pytest.fixture
def audio_upload(make_upload):
    return make_upload("audio", b"original audio bytes", duration=2.0, has_audio=True)


@pytest.fixture
def image_upload(make_upload):
    return make_upload("image", _jpeg((0, 0, 200)))


async def _run(session, car, text, uploads):
    return [x async for x in run_turn(session, car, text, uploads)]


# --- run_turn -------------------------------------------------------------------

async def test_video_turn_emits_media_stage_and_sends_frames(fake_all, session, car, video_upload):
    events = await _run(session, car, None, [video_upload])
    assert events[0] == PROCESSING and events[1] == SEARCHING and events[-1][0] == "done"
    sent = fake_all.llm.prompt[-1]["content"]
    assert sum(b["type"] == "image_url" for b in sent) == 8
    assert "belt squeal (0.70)" in sent[0]["text"]
    assert _image_blocks(fake_all.llm.prompt) == FRAMES
    fake_all.media.extract_frames.assert_awaited_once_with(video_upload.path, 3.0, 8)
    fake_all.hearing.transcribe.assert_awaited_once_with(video_upload.path)
    assert fake_all.llm.calls == 2


async def test_silent_video_skips_hearing(fake_all, session, car, silent_video_upload):
    _ = await _run(session, car, "leak under engine", [silent_video_upload])
    fake_all.hearing.transcribe.assert_not_called()
    fake_all.hearing.tag_sounds.assert_not_called()
    fake_all.media.audio_pcm.assert_not_called()
    assert len(_image_blocks(fake_all.llm.prompt)) == 8
    user_msg = fake_all.store.messages(session["id"])[0]
    assert user_msg["observations"] is None and user_msg["media"][0]["kind"] == "video"


async def test_media_paths_saved_on_user_message(fake_all, session, car, video_upload):
    _ = await _run(session, car, None, [video_upload])
    user_msg = fake_all.store.messages(session["id"])[0]
    assert user_msg["media"][0]["kind"] == "video" and user_msg["observations"]["sounds"]
    assert user_msg["content"] == ""


async def test_image_turn_sends_jpeg_bytes_and_skips_hearing(fake_all, session, car, make_upload):
    images = [make_upload("image", _jpeg((0, 0, 200))), make_upload("image", _jpeg((0, 200, 0)))]
    out = await _run(session, car, "what is this puddle", images)
    assert out[0] == PROCESSING and out[-1][0] == "done"
    assert _image_blocks(fake_all.llm.prompt) == [u.path.read_bytes() for u in images]
    fake_all.hearing.transcribe.assert_not_called()
    fake_all.hearing.tag_sounds.assert_not_called()
    fake_all.media.extract_frames.assert_not_called()
    user_msg = fake_all.store.messages(session["id"])[0]
    assert [m["kind"] for m in user_msg["media"]] == ["image", "image"] and user_msg["observations"] is None


async def test_audio_turn_tags_and_transcribes(fake_all, session, car, audio_upload):
    out = await _run(session, car, None, [audio_upload])
    assert out[-1][0] == "done"
    fake_all.media.audio_pcm.assert_awaited_once_with(audio_upload.path, sr=48000)
    assert fake_all.hearing.tag_sounds.await_args.args[0] is PCM
    assert fake_all.hearing.tag_sounds.await_args.kwargs == {"sr": 48000}
    fake_all.hearing.transcribe.assert_awaited_once_with(audio_upload.path)
    fake_all.media.extract_frames.assert_not_called()
    assert _image_blocks(fake_all.llm.prompt) == []
    assert TRANSCRIPT in fake_all.llm.prompt[-1]["content"][0]["text"]
    user_msg = fake_all.store.messages(session["id"])[0]
    assert user_msg["observations"] == {"transcript": TRANSCRIPT, "sounds": [{"label": "belt squeal", "score": 0.7}]}
    assert user_msg["media"][0]["kind"] == "audio"


async def test_kb_query_includes_text_transcript_and_sound_labels(fake_all, session, car, audio_upload):
    await _run(session, car, "squeal at startup", [audio_upload])
    assert [q for q, _, _ in fake_all.kb.queries] == [f"squeal at startup {TRANSCRIPT} belt squeal"]


async def test_kb_query_from_media_alone(fake_all, session, car, audio_upload):
    await _run(session, car, None, [audio_upload])
    assert [q for q, _, _ in fake_all.kb.queries] == [f"{TRANSCRIPT} belt squeal"]


async def test_no_text_and_no_sound_skips_search(fake_all, session, car, silent_video_upload):
    out = await _run(session, car, None, [silent_video_upload])
    assert fake_all.kb.queries == [] and out[-1][0] == "done"


@pytest.mark.parametrize("failure", [MediaError(detail="ffmpeg: /srv/tmp/abc.bin: bad"), RuntimeError("model died")])
async def test_media_failure_is_error_event_and_keeps_user_message(fake_all, session, car, video_upload, failure):
    fake_all.hearing.tag_sounds.side_effect = failure
    out = await _run(session, car, "listen to this", [video_upload])
    assert out == [PROCESSING, ("error", {"message": MEDIA_ERROR})]
    msgs = fake_all.store.messages(session["id"])
    assert [(m["role"], m["content"]) for m in msgs] == [("user", "listen to this")]
    assert msgs[0]["observations"] is None
    # The original still reached Storage (it uploads alongside processing), so the ref is kept.
    assert msgs[0]["media"] == [{"kind": "video", "path": f"{session['user_id']}/{session['id']}/{msgs[0]['id']}/0.mp4"}]
    assert fake_all.llm.calls == 0
    assert fake_all.store.rows[session["id"]]["state"] == {}


async def test_failed_job_cancels_its_siblings_before_the_turn_ends(fake_all, session, car, video_upload):
    fake_all.hearing.tag_sounds.side_effect = RuntimeError("CLAP failed to load")
    seen = {"started": 0, "cancelled": 0}

    async def slow(*args):
        seen["started"] += 1
        try:
            await asyncio.Event().wait()  # never set: only cancellation ends it
        except asyncio.CancelledError:
            seen["cancelled"] += 1
            raise

    fake_all.hearing.transcribe.side_effect = slow
    fake_all.media.extract_frames.side_effect = slow
    out = await asyncio.wait_for(_run(session, car, None, [video_upload]), 2)
    assert out == [PROCESSING, ("error", {"message": MEDIA_ERROR})]
    assert seen == {"started": 2, "cancelled": 2}  # transcribe and frames, both cancelled before run_turn returned


async def test_media_failure_logs_the_leaf_type_and_ffmpeg_detail(fake_all, session, car, video_upload, monkeypatch):
    log = SimpleNamespace(error=AsyncMock(), warning=AsyncMock())
    monkeypatch.setattr(diagnosis, "logger", log)
    fake_all.media.extract_frames.side_effect = MediaError(detail="ffmpeg: moov atom not found")
    await _run(session, car, None, [video_upload])
    [message] = [c.args[0] for c in log.error.await_args_list]
    assert message == "media processing failed: MediaError: ffmpeg: moov atom not found"


async def test_video_with_no_frames_and_no_audio_is_a_media_error(fake_all, session, car, silent_video_upload):
    fake_all.media.extract_frames.return_value = []  # every frame failed to decode
    out = await _run(session, car, None, [silent_video_upload])
    assert out == [PROCESSING, ("error", {"message": MEDIA_ERROR})]
    assert fake_all.llm.calls == 0


async def test_video_with_no_frames_but_audio_still_runs(fake_all, session, car, video_upload):
    fake_all.media.extract_frames.return_value = []
    out = await _run(session, car, None, [video_upload])
    assert out[-1][0] == "done" and _image_blocks(fake_all.llm.prompt) == []


async def test_storage_frames_and_hearing_run_concurrently(fake_all, session, car, video_upload, monkeypatch):
    # A ring: each job waits for an event only the next one sets, so any serial order deadlocks.
    stored, framed, heard = asyncio.Event(), asyncio.Event(), asyncio.Event()
    real_upload = fake_all.bucket.upload_file

    async def upload(*args):
        stored.set()
        await heard.wait()
        return await real_upload(*args)

    async def frames(*args):
        framed.set()
        await stored.wait()
        return FRAMES

    async def transcribe(*args):
        heard.set()
        await framed.wait()
        return TRANSCRIPT

    monkeypatch.setattr(fake_all.bucket, "upload_file", upload)
    fake_all.media.extract_frames.side_effect = frames
    fake_all.hearing.transcribe.side_effect = transcribe
    out = await asyncio.wait_for(_run(session, car, None, [video_upload]), 2)
    assert out[-1][0] == "done" and len(_image_blocks(fake_all.llm.prompt)) == 8


async def test_media_only_turn_uses_a_placeholder_for_empty_content(fake_all, fake_store, car, audio_upload):
    history = [
        {"id": "a", "role": "user", "content": "", "media": [], "observations": None, "created_at": "x"},
        {"id": "b", "role": "assistant", "content": "sounds like a belt", "media": [], "observations": None, "created_at": "y"},
    ]
    session = fake_store.add("u", car["id"], messages=history)
    await _run(session, car, None, [audio_upload])
    assert fake_all.llm.prompt[1] == {"role": "user", "content": "[attached media]"}
    assert all(m["content"] for m in fake_all.llm.conversation)
    assert fake_all.llm.conversation[0]["content"] == "[attached media]"
    assert fake_all.llm.conversation[2] == {"role": "user", "content": "[attached media]"}
    assert fake_store.messages(session["id"])[2]["content"] == ""  # stored as sent; only the prompt is padded


async def test_storage_failure_does_not_kill_the_turn(fake_all, session, car, video_upload):
    fake_all.bucket.error = RuntimeError("Bucket not found")
    out = await _run(session, car, None, [video_upload])
    assert out[-1][0] == "done"
    assert len(_image_blocks(fake_all.llm.prompt)) == 8
    user_msg = fake_all.store.messages(session["id"])[0]
    assert user_msg["media"] == [] and user_msg["observations"]["sounds"]


async def test_storage_paths_follow_user_session_message_scheme(fake_all, session, car, make_upload, video_upload):
    sid, uid = session["id"], session["user_id"]
    await _run(session, car, None, [video_upload])
    msg_id = fake_all.store.messages(sid)[0]["id"]
    path = f"{uid}/{sid}/{msg_id}/0.mp4"
    assert fake_all.bucket.objects == {("media", path): b"original video bytes"}
    assert fake_all.store.messages(sid)[0]["media"] == [{"kind": "video", "path": path}]

    images = [make_upload("image", _jpeg((1, 2, 3))), make_upload("image", _jpeg((4, 5, 6)))]
    await _run(await fake_all.store.get_for_user(sid, uid), car, "photos", images)
    msg_id = fake_all.store.messages(sid)[2]["id"]
    assert fake_all.store.messages(sid)[2]["media"] == [
        {"kind": "image", "path": f"{uid}/{sid}/{msg_id}/0.jpg"},
        {"kind": "image", "path": f"{uid}/{sid}/{msg_id}/1.jpg"},
    ]
    assert fake_all.bucket.objects[("media", f"{uid}/{sid}/{msg_id}/1.jpg")] == images[1].path.read_bytes()
    assert {ct for _, p, ct in fake_all.bucket.uploads if p.endswith(".jpg")} == {"image/jpeg"}


async def test_audio_original_stored_with_audio_label(fake_all, session, car, audio_upload):
    await _run(session, car, None, [audio_upload])
    msg = fake_all.store.messages(session["id"])[0]
    assert msg["media"] == [{"kind": "audio", "path": f"{session['user_id']}/{session['id']}/{msg['id']}/0.m4a"}]


async def test_empty_reply_is_an_error_and_saves_nothing_more(fake_all, session, car):
    fake_all.llm.reply = "   "
    out = await _run(session, car, "fan clicks", [])
    assert out == [SEARCHING, ("error", {"message": LLM_ERROR})]
    assert fake_all.llm.calls == 1  # no state call
    row = fake_all.store.rows[session["id"]]
    assert [m["role"] for m in row["messages"]] == ["user"]
    assert row["state"] == {} and row["title"] == "New diagnosis"


@pytest.mark.parametrize("with_media", [True, False])
async def test_user_message_saved_before_first_event_is_consumed(fake_all, session, car, image_upload, with_media):
    gen = run_turn(session, car, "look at this", [image_upload] if with_media else [])
    first = await anext(gen)
    assert first == (PROCESSING if with_media else SEARCHING)
    await gen.aclose()
    assert [(m["role"], m["content"]) for m in fake_all.store.messages(session["id"])] == [("user", "look at this")]
    assert fake_all.llm.calls == 0


# --- send route -------------------------------------------------------------------

@pytest.fixture
def turn_dirs(monkeypatch):
    """Every temp dir the send route makes."""
    made: list[Path] = []
    real = tempfile.mkdtemp

    def spy(*args, **kwargs):
        d = real(*args, **kwargs)
        made.append(Path(d))
        return d

    monkeypatch.setattr(session_route.tempfile, "mkdtemp", spy)
    return made


@pytest.fixture
def fake_mp4(monkeypatch):
    """A text file named .mp4. ffprobe is faked: it rejects the file, as the real one does."""

    async def probe(path):
        raise MediaError(detail=f"{path}: Invalid data found when processing input")

    monkeypatch.setattr(media, "probe", probe)
    return ("fake.mp4", b"this is not a video, just text\n", "video/mp4")


def _url(session_id):
    return f"/api/v1/sessions/{session_id}/messages"


def test_validation_error_is_json_not_stream(client, auth_headers, existing_session, fake_mp4, fake_llm, turn_dirs):
    r = client.post(_url(existing_session["id"]), files={"files": fake_mp4}, headers=auth_headers)
    assert r.status_code == 415 and r.headers["content-type"].startswith("application/json")
    assert fake_llm.calls == 0
    assert existing_session["id"] not in session_route._in_flight
    assert len(turn_dirs) == 1 and not turn_dirs[0].exists()
    r = client.post(_url(existing_session["id"]), data={"text": "fan clicks"}, headers=auth_headers)
    assert r.status_code == 200 and _parse(r.text)[-1][0] == "done"


def test_photo_turn_over_http(client, auth_headers, existing_session, fake_all, turn_dirs):
    sid, uid = existing_session["id"], existing_session["user_id"]
    photo = _jpeg((10, 120, 10))
    r = client.post(_url(sid), data={"text": "puddle"}, files={"files": ("p.jpg", photo, "image/jpeg")}, headers=auth_headers)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache, no-store" and r.headers["x-accel-buffering"] == "no"
    events = _parse(r.text)
    assert events[:2] == [PROCESSING, SEARCHING] and events[-1][0] == "done"
    assert len(_image_blocks(fake_all.llm.prompt)) == 1
    user_msg = fake_all.store.messages(sid)[0]
    assert user_msg["media"] == [{"kind": "image", "path": f"{uid}/{sid}/{user_msg['id']}/0.jpg"}]
    assert sid not in session_route._in_flight
    assert len(turn_dirs) == 1 and not turn_dirs[0].exists()


def test_tmpdir_removed_after_llm_error(client, auth_headers, existing_session, fake_all, turn_dirs):
    fake_all.llm.fail_stream_after = 1
    r = client.post(
        _url(existing_session["id"]), files={"files": ("p.jpg", _jpeg(), "image/jpeg")}, headers=auth_headers
    )
    assert _parse(r.text)[-1] == ("error", {"message": LLM_ERROR})
    assert len(turn_dirs) == 1 and not turn_dirs[0].exists()
    assert existing_session["id"] not in session_route._in_flight


def test_text_only_send_makes_no_tmpdir(client, auth_headers, existing_session, fake_llm, turn_dirs):
    r = client.post(_url(existing_session["id"]), data={"text": "fan clicks"}, headers=auth_headers)
    assert r.status_code == 200 and turn_dirs == []


async def test_tmpdir_removed_when_client_disconnects(existing_session, car, fake_all, make_upload):
    sid = existing_session["id"]
    tmpdir = tempfile.mkdtemp(prefix="autocare-test-")
    upload = make_upload("image", _jpeg(), folder=Path(tmpdir))
    session_route._in_flight.add(sid)
    stream, _ = session_route._events(existing_session, car, None, [upload], tmpdir)
    assert (await anext(stream)).startswith("event: stage")
    await stream.aclose()
    assert sid not in session_route._in_flight and not Path(tmpdir).exists()


async def test_disconnect_during_processing_cancels_jobs_and_releases(existing_session, car, fake_all, make_upload):
    sid = existing_session["id"]
    tmpdir = tempfile.mkdtemp(prefix="autocare-test-")
    upload = make_upload("video", b"clip", duration=3.0, has_audio=False, folder=Path(tmpdir))
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def frames(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    fake_all.media.extract_frames.side_effect = frames
    session_route._in_flight.add(sid)
    stream, _ = session_route._events(existing_session, car, None, [upload], tmpdir)
    assert (await anext(stream)).startswith("event: stage")
    pending = asyncio.ensure_future(anext(stream))  # now inside the media gather
    await asyncio.wait_for(started.wait(), 2)
    pending.cancel()  # what Starlette does to the response task when the client goes away
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cancelled.is_set()
    assert sid not in session_route._in_flight and not Path(tmpdir).exists()
    assert [m["role"] for m in fake_all.store.messages(sid)] == ["user"]


def test_tmpdir_removed_if_stream_never_starts(existing_session, car, make_upload):
    sid = existing_session["id"]
    tmpdir = tempfile.mkdtemp(prefix="autocare-test-")
    upload = make_upload("image", _jpeg(), folder=Path(tmpdir))
    session_route._in_flight.add(sid)
    stream, _ = session_route._events(existing_session, car, None, [upload], tmpdir)
    del stream
    gc.collect()
    assert sid not in session_route._in_flight and not Path(tmpdir).exists()


# --- Storage --------------------------------------------------------------------

class _FakeStorage:
    """Mimics supabase storage list(): one folder level per call, sub-folders have id None, paged."""

    def __init__(self, files):
        self.files = set(files)
        self.removed: list[list[str]] = []
        self.uploaded: list[tuple] = []

    def from_(self, bucket):
        self.bucket = bucket
        return self

    def list(self, path=None, options=None):
        path = (path or "").strip("/")
        level: dict[str, str | None] = {}
        for f in sorted(self.files):
            if f.startswith(path + "/"):
                head, sep, _ = f[len(path) + 1 :].partition("/")
                level[head] = None if sep else f"id-{head}"
        entries = [{"name": n, "id": i} for n, i in sorted(level.items())]
        options = options or {}
        offset, limit = options.get("offset", 0), options.get("limit", 100)
        return entries[offset : offset + limit]

    def remove(self, paths):
        self.removed.append(list(paths))
        self.files -= set(paths)
        return [{"name": p} for p in paths]

    def upload(self, path, file, file_options=None):
        self.uploaded.append((self.bucket, path, file.read(), file_options))


@pytest.fixture
def fake_storage(monkeypatch):
    def install(files=()):
        storage = _FakeStorage(files)
        monkeypatch.setattr(SupabaseBucketManager, "client", property(lambda self: SimpleNamespace(storage=storage)))
        return storage

    return install


async def test_delete_prefix_removes_nested_files_in_batches(fake_storage):
    keep = "u1/other-session/m1/0.jpg"
    mine = ["u1/s1/m1/0.jpg", "u1/s1/m1/1.jpg", "u1/s1/m2/0.mp4"] + [f"u1/s1/m3/{i}.jpg" for i in range(150)]
    storage = fake_storage([*mine, keep])
    n = await SupabaseBucketManager().delete_prefix("media", "u1/s1/")
    assert n == len(mine)
    assert storage.files == {keep} and storage.bucket == "media"
    assert all(len(batch) <= 100 for batch in storage.removed)
    assert sorted(p for batch in storage.removed for p in batch) == sorted(mine)


async def test_delete_prefix_with_nothing_there_removes_nothing(fake_storage):
    storage = fake_storage(["u1/s2/m1/0.jpg"])
    assert await SupabaseBucketManager().delete_prefix("media", "u1/s1/") == 0
    assert storage.removed == []


async def test_upload_file_sends_content_type(fake_storage, tmp_path):
    storage = fake_storage()
    src = tmp_path / "x.jpg"
    src.write_bytes(b"jpeg")
    await SupabaseBucketManager().upload_file("media", str(src), "u1/s1/m1/0.jpg", content_type="image/jpeg")
    assert storage.uploaded == [("media", "u1/s1/m1/0.jpg", b"jpeg", {"content-type": "image/jpeg"})]
