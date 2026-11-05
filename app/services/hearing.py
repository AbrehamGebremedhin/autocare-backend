"""Local hearing: speech transcription (faster-whisper) and car-sound tagging (CLAP)."""

import asyncio
import threading
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.schemas.session import SoundTag
from app.services.media import audio_pcm

CAR_SOUND_LABELS: list[str] = [
    # catch-alls (kept in the softmax so fault labels compete against them)
    "normal engine idle",
    "human speech",
    "background noise",
    # engine
    "engine knocking",
    "valve ticking",
    "rough idle or engine misfire",
    "engine backfiring",
    "exhaust leak puffing",
    "loud exhaust rumble",
    "turbo whistle or whine",
    # belts and accessories
    "belt squeal",
    "alternator whining",
    "power steering pump whining",
    "water pump grinding",
    "air conditioning compressor rattle",
    # leaks
    "hissing from a vacuum or coolant leak",
    # starting
    "starter motor clicking",
    "starter motor grinding",
    "engine cranking slowly",
    # brakes
    "brake grinding",
    "brake squeal",
    # suspension, steering, drivetrain
    "clunking over bumps",
    "rattling heat shield",
    "CV joint clicking when turning",
    "wheel bearing humming",
    "suspension creaking",
    "tire thumping",
    "transmission whining",
    "differential howling",
    "fan motor buzzing",
]

MAX_TRANSCRIPT_CHARS = 2000  # the transcript feeds an LLM prompt
CLAP_SR = 48000
WINDOW_SECONDS = 10
MIN_TAIL_SECONDS = 2
_DROPPED = {"human speech", "background noise"}

_lock = threading.Lock()
_whisper_model = None
_clap_model = None


def _whisper():
    global _whisper_model
    with _lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel

            _whisper_model = WhisperModel(get_settings().WHISPER_MODEL, device="cpu", compute_type="int8")
        return _whisper_model


def _clap():
    global _clap_model
    with _lock:
        if _clap_model is None:
            from transformers import ClapModel, ClapProcessor

            name = get_settings().CLAP_MODEL
            _clap_model = (ClapModel.from_pretrained(name).eval(), ClapProcessor.from_pretrained(name))
        return _clap_model


WHISPER_SR = 16000


def _transcribe_sync(pcm: np.ndarray) -> str:
    # Pass the already-capped array: given a path, faster-whisper would decode the whole file.
    # beam_size=1 (greedy): several times faster on CPU, plenty for short diagnostic clips.
    segments, _ = _whisper().transcribe(pcm, vad_filter=True, beam_size=1)
    limit = get_settings().MAX_MEDIA_SECONDS
    parts: list[str] = []
    size = 0
    for seg in segments:  # lazy generator; second guard besides the capped decode
        if seg.start > limit or size >= MAX_TRANSCRIPT_CHARS:
            break
        text = seg.text.strip()
        if text:
            parts.append(text)
            size += len(text) + 1
    return " ".join(parts)[:MAX_TRANSCRIPT_CHARS]


async def transcribe(path: Path) -> str:
    """Transcribe speech in a media file ("" if none).

    Raises media.MediaError if the file has no decodable audio; callers check has_audio first.
    """
    pcm = await audio_pcm(path, sr=WHISPER_SR)  # ffmpeg decode capped at MAX_MEDIA_SECONDS
    return await asyncio.to_thread(_transcribe_sync, pcm)


def _resample(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Plain linear interpolation to 48 kHz; good enough for tagging (no torchaudio/librosa)."""
    if sr == CLAP_SR or pcm.size == 0:
        return pcm
    n = round(pcm.size * CLAP_SR / sr)
    x = np.linspace(0, pcm.size - 1, num=n)
    return np.interp(x, np.arange(pcm.size), pcm).astype(np.float32)


def _windows(pcm: np.ndarray, sr: int) -> list[np.ndarray]:
    """10 s windows; a partial tail counts only if >= 2 s; a short clip is one window."""
    if pcm.size == 0:
        return []
    size = WINDOW_SECONDS * sr
    wins = [pcm[i : i + size] for i in range(0, pcm.size, size)]
    if len(wins) > 1 and wins[-1].size < MIN_TAIL_SECONDS * sr:
        wins.pop()
    return wins


def _tag_sync(pcm: np.ndarray, sr: int, top_k: int) -> list[SoundTag]:
    import torch

    wins = _windows(_resample(pcm, sr), CLAP_SR)
    if not wins:
        return []
    model, processor = _clap()
    # Plain labels as the text prompts: CLAP matches short natural phrases well and this
    # keeps the returned label identical to CAR_SOUND_LABELS.
    # Tokenise and featurise separately: processor(..., padding=True) would override the
    # feature extractor's default "repeatpad" and zero-pad clips shorter than 10 s.
    text = processor.tokenizer(CAR_SOUND_LABELS, padding=True, return_tensors="pt")
    audio = processor.feature_extractor(wins, sampling_rate=CLAP_SR, return_tensors="pt")
    with torch.inference_mode():
        probs = model(**text, **audio).logits_per_audio.softmax(dim=-1)  # (windows, labels)
    best = probs.max(dim=0).values.tolist()
    floor = get_settings().SOUND_MIN_SCORE
    tags = [
        SoundTag(label=lbl, score=float(s))
        for lbl, s in zip(CAR_SOUND_LABELS, best)
        if lbl not in _DROPPED and s >= floor
    ]
    tags.sort(key=lambda t: t.score, reverse=True)
    return tags[:top_k]


async def tag_sounds(pcm: np.ndarray, sr: int = CLAP_SR, top_k: int = 3) -> list[SoundTag]:
    return await asyncio.to_thread(_tag_sync, pcm, sr, top_k)


def prefetch() -> None:
    """Download both models into the HF / CTranslate2 cache (Docker build step)."""
    _whisper()
    _clap()


if __name__ == "__main__":
    prefetch()
