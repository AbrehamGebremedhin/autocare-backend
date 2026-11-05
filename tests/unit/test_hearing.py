import types

import numpy as np
import pytest
import torch

from app.core.config import get_settings
from app.services import hearing
from app.services.hearing import CAR_SOUND_LABELS, _windows, tag_sounds, transcribe

SR = 48000
LABELS = CAR_SOUND_LABELS
N = len(LABELS)


def _probs_to_logits(probs_by_label: dict[str, float], rest: float) -> torch.Tensor:
    """Logits whose softmax over LABELS gives the requested probabilities;
    the leftover mass is spread evenly over the unnamed labels."""
    others = [lbl for lbl in LABELS if lbl not in probs_by_label]
    p = torch.empty(N, dtype=torch.float64)
    for i, lbl in enumerate(LABELS):
        p[i] = probs_by_label.get(lbl, rest / len(others))
    assert p.sum().item() == pytest.approx(1.0)
    return torch.log(p).float()


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, *, text, audios, sampling_rate, return_tensors, padding):
        self.calls.append(types.SimpleNamespace(text=text, audios=audios, sr=sampling_rate))
        return {"n": len(audios)}


class _FakeModel:
    def __init__(self, per_window):
        self.per_window = per_window  # callable(window_index) -> logits row
        self.seen = 0

    def __call__(self, **inputs):  # the real call is model(**inputs)
        n = inputs["n"]
        rows = torch.stack([self.per_window(self.seen + i) for i in range(n)])
        self.seen += n
        return types.SimpleNamespace(logits_per_audio=rows)


@pytest.fixture
def fake_clap(mocker):
    def row(i):
        if i == 1:  # window 2 only
            return _probs_to_logits({"belt squeal": 0.7, "background noise": 0.2}, rest=0.1)
        return _probs_to_logits({"background noise": 0.9}, rest=0.1)

    model = _FakeModel(row)
    proc = _FakeProcessor()
    mocker.patch("app.services.hearing._clap", return_value=(model, proc))
    return proc


@pytest.fixture
def fake_clap_flat(mocker):
    model = _FakeModel(lambda i: torch.zeros(N))
    mocker.patch("app.services.hearing._clap", return_value=(model, _FakeProcessor()))


async def test_tag_sounds_max_over_windows_and_threshold(fake_clap):
    tags = await tag_sounds(np.zeros(SR * 25, dtype=np.float32))
    assert tags[0].label == "belt squeal" and tags[0].score == pytest.approx(0.7)
    assert all(t.label not in ("human speech", "background noise") for t in tags)
    assert all(t.score >= get_settings().SOUND_MIN_SCORE for t in tags)
    assert len(fake_clap.calls[0].audios) == 3
    assert fake_clap.calls[0].sr == SR
    assert fake_clap.calls[0].text == LABELS  # plain labels are the text prompts


async def test_tag_sounds_empty_when_nothing_clears_threshold(fake_clap_flat):
    assert await tag_sounds(np.zeros(SR * 5, dtype=np.float32)) == []


async def test_tag_sounds_resamples_to_48k(fake_clap):
    await tag_sounds(np.zeros(16000 * 5, dtype=np.float32), sr=16000)
    [w] = fake_clap.calls[0].audios
    assert len(w) == SR * 5


@pytest.mark.parametrize(
    "seconds, expected",
    [(25, 3), (20, 2), (12, 2), (11, 1), (10.5, 1), (10, 1), (1, 1), (60, 6)],
)
def test_windows_rules(seconds, expected):
    wins = _windows(np.zeros(int(SR * seconds), dtype=np.float32), SR)
    assert len(wins) == expected
    assert sum(len(w) for w in wins) <= int(SR * seconds)


def test_label_list_shape():
    assert 28 <= len(CAR_SOUND_LABELS) <= 32
    assert len(set(CAR_SOUND_LABELS)) == len(CAR_SOUND_LABELS)
    for catch_all in ("normal engine idle", "human speech", "background noise"):
        assert catch_all in CAR_SOUND_LABELS


class _Seg:
    def __init__(self, text, start=0.0):
        self.text, self.start = text, start


class _FakeWhisper:
    def __init__(self, segs):
        self.segs = segs
        self.kwargs = None

    def transcribe(self, path, **kw):
        self.kwargs = kw
        return iter(self.segs), None


async def test_transcribe_joins_segments(mocker, tmp_path):
    fake = _FakeWhisper([_Seg(" my car "), _Seg("makes a noise  ")])
    mocker.patch("app.services.hearing._whisper", return_value=fake)
    assert await transcribe(tmp_path / "a.mp4") == "my car makes a noise"
    assert fake.kwargs["vad_filter"] is True and fake.kwargs["beam_size"] == 1


async def test_transcribe_empty_when_no_speech(mocker, tmp_path):
    mocker.patch("app.services.hearing._whisper", return_value=_FakeWhisper([]))
    assert await transcribe(tmp_path / "a.mp4") == ""


async def test_transcribe_stops_after_max_media_seconds(mocker, tmp_path):
    consumed = []

    def endless():
        t = 0.0
        while True:
            consumed.append(t)
            yield _Seg("blah", start=t)
            t += 5.0

    fake = _FakeWhisper([])
    fake.transcribe = lambda path, **kw: (endless(), None)
    mocker.patch("app.services.hearing._whisper", return_value=fake)
    out = await transcribe(tmp_path / "a.mp4")
    assert out
    assert max(consumed) <= get_settings().MAX_MEDIA_SECONDS + 5


async def test_transcribe_caps_length(mocker, tmp_path):
    mocker.patch("app.services.hearing._whisper", return_value=_FakeWhisper([_Seg("x" * 5000)]))
    assert len(await transcribe(tmp_path / "a.mp4")) == hearing.MAX_TRANSCRIPT_CHARS


@pytest.mark.slow
async def test_real_models_on_sine(tmp_path):  # opt-in: uv run pytest -m slow
    import subprocess

    from app.services.media import audio_pcm

    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
         "-f", "lavfi", "-i", "testsrc=duration=12:size=64x48:rate=5", "-shortest",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )  # fmt: skip
    assert isinstance(await transcribe(clip), str)
    tags = await tag_sounds(await audio_pcm(clip))
    assert isinstance(tags, list)
