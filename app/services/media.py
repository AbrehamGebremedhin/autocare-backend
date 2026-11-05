from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Upload:
    kind: Literal["image", "audio", "video"]
    path: Path
    mime: str
    duration: float | None
    has_audio: bool
