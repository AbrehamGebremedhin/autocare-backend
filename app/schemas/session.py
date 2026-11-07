from typing import Literal

from pydantic import BaseModel, Field, field_validator

TITLE_MAX_CHARS = 60  # the state prompt asks for at most 60; enforced here because the model may not comply


class SuspectedCause(BaseModel):
    cause: str
    likelihood: float = Field(ge=0, le=1)
    evidence: str = ""
    next_check: str = ""


class DiagnosisState(BaseModel):
    suspected_causes: list[SuspectedCause] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    title: str = ""

    @field_validator("title")
    @classmethod
    def _cap_title(cls, title: str) -> str:
        return title.strip()[:TITLE_MAX_CHARS].rstrip()


class SoundTag(BaseModel):
    label: str
    score: float


class Observations(BaseModel):
    transcript: str = ""
    sounds: list[SoundTag] = Field(default_factory=list)


class MediaRef(BaseModel):
    kind: Literal["image", "audio", "video"]
    path: str


class Message(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    media: list[MediaRef] = Field(default_factory=list)
    observations: Observations | None = None
    created_at: str
