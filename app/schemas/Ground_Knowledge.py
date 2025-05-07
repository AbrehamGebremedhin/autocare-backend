from pydantic import BaseModel, Field
from typing import List, Optional

class GroundKnowledgeBase(BaseModel):
    id: Optional[str] = None
    book_title: str
    content_chunk: str  # The text chunk from the book
    vector: List[float]  # The vector representation of the chunk
    page_number: Optional[int] = None
    metadata: Optional[dict] = None  # Any additional metadata

    class Config:
        orm_mode = True
