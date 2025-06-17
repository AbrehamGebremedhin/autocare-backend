from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class Diagnosis_History(BaseModel):
    id: Optional[str] = Field(default=None, primary_key=True, unique=True)
    user_id: str
    session_data: Dict[str, Any]
    timestamp: datetime = datetime.now()

    class Config:
        from_attributes = True
