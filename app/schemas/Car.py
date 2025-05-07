from pydantic import BaseModel
from typing import Optional

class CarBase(BaseModel):
    id: Optional[str] = None
    make: str
    model: str
    year: int
    owner_manual_url: Optional[str] = None  # URL or path to owner manual PDF in Supabase
    service_manual_url: Optional[str] = None  # URL or path to service manual PDF in Supabase

    class Config:
        orm_mode = True
