from pydantic import BaseModel
from typing import Optional, List

class CarBase(BaseModel):
    id: Optional[str] = None
    make: str
    model: str
    year: int
    owner_manual_url: Optional[str] = None  # URL or path to owner manual PDF in Supabase
    service_manual_url: Optional[str] = None  # URL or path to service manual PDF in Supabase
    car_guide_links: Optional[List[str]] = None  # List of additional car guide links

    class Config:
        orm_mode = True
