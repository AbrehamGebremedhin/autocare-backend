from pydantic import BaseModel, Field
from typing import Optional, List

class CarBase(BaseModel):
    id: Optional[str] = Field(default=None)
    make: str
    model: str
    year: int
    owner_manual_url: Optional[str] = None  # URL or path to owner manual PDF in Supabase
    service_manual_url: Optional[str] = None  # URL or path to service manual PDF in Supabase
    car_guide_links: Optional[List[str]] = None  # List of additional car guide links
    is_vectorized: bool = Field(default=False, description="Whether manual text is vectorized in Milvus")
    vector_chunk_count: int = Field(default=0, description="Number of chunks in Milvus")

    class Config:
        from_attributes = True
