from pydantic import BaseModel, Field


class SuspectedCause(BaseModel):
    cause: str
    likelihood: float = Field(ge=0, le=1)
    evidence: str = ""
    next_check: str = ""


class DiagnosisState(BaseModel):
    suspected_causes: list[SuspectedCause] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    title: str = ""
