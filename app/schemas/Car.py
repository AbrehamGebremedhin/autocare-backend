from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CarGuideLink(BaseModel):
    link: str
    summary: Optional[str] = None

class CarBase(BaseModel):
    id: Optional[str] = Field(default=None)
    make: str
    model: str
    year: int
    owner_manual_url: Optional[str] = None  # URL or path to owner manual PDF in Supabase
    service_manual_url: Optional[str] = None  # URL or path to service manual PDF in Supabase
    car_guide_links: Optional[List[CarGuideLink]] = None  # List of additional car guide links with summaries

    class Config:
        from_attributes = True
