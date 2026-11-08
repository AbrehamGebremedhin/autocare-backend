"""Vehicle artwork descriptor: the exact contract the mobile client accepts, nothing else."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Artwork(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["flat_webp"] = "flat_webp"
    width: Literal[1024] = 1024
    height: Literal[420] = 420
    baseline_y: Literal[366] = 366
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    download_url: str = Field(pattern=r"^https://.+")
    expires_at: str


class VisualOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant_key: str
    label: str


class VisualDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    state: Literal["ready", "pending", "fallback"]
    variant_key: str | None = None
    style_version: str | None = None
    preset: Literal["garage_roll_v1"] | None = "garage_roll_v1"
    reason: Literal["unknown_variant", "choose_variant", "temporarily_unavailable"] | None = None
    retry_after_ms: int | None = None
    options: list[VisualOption] | None = None
    artwork: Artwork | None = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)
