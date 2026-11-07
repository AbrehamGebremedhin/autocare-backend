from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Request limits for car input. Postgres text cannot hold NUL, so it is refused up front (422, not a 500).
NO_NUL = r"^[^\x00]*$"
MAX_CAR_NAME_CHARS = 64  # make and model
MAX_CAR_ID_CHARS = 200  # ids are "<make>-<model>-<year>"
MIN_CAR_YEAR = 1900

class CarGuideLink(BaseModel):
    link: str
    summary: Optional[str] = None

class CarBase(BaseModel):
    id: Optional[str] = Field(default=None)
    make: str
    model: str
    year: int
    owner_manual_url: Optional[str] = None  # URL of the owner manual PDF
    service_manual_url: Optional[str] = None  # URL of the service manual PDF
    car_guide_links: Optional[List[CarGuideLink]] = None  # List of additional car guide links with summaries
    kb_status: str = "pending"  # pending | ready | failed | no_manual
    kb_passage_count: int = 0

    class Config:
        from_attributes = True
