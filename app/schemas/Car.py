from pydantic import BaseModel, Field
from typing import Optional, List

class CarBase(BaseModel):
    id: Optional[str] = Field(default=None)
    make: str
    model: str
    year: int
    vector:  List[float]  # The vector representation of the chunk
    owner_manual_url: Optional[str] = None  # URL or path to owner manual PDF in Supabase
    service_manual_url: Optional[str] = None  # URL or path to service manual PDF in Supabase
    car_guide_links: Optional[List[str]] = None  # List of additional car guide links

    class Config:
        from_attributes = True
