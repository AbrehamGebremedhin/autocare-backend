from datetime import datetime
from typing import Optional, Any, Dict
from .message_types import MessageType, MessageSource

class MessageFormatter:
    @staticmethod
    def format(
        *,
        type: MessageType,
        source: MessageSource,
        content: str,
        session_id: Optional[str] = None,
        progress: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        msg = {
            "type": type.value,
            "source": source.value,
            "content": content,
            "timestamp": timestamp or datetime.utcnow().isoformat() + "Z",
        }
        if session_id is not None:
            msg["session_id"] = session_id
        if progress is not None:
            msg["progress"] = progress
        if details is not None:
            msg["details"] = details
        return msg
