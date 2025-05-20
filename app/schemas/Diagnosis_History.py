from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class Diagnosis_History(BaseModel):
    user_id: str
    session_data: Dict[str, Any]
    timestamp: datetime = datetime.now()

    class Config:
        orm_mode = True
