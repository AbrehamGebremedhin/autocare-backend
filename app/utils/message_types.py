from enum import Enum

class MessageType(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    PROGRESS = "progress"
    STAGE = "stage"
    RESULT = "result"
    DEBUG = "debug"

class MessageSource(str, Enum):
    CHAT_SERVICE = "chat_service"
    DIAGNOSTIC_AGENT = "diagnostic_agent"
    SYMPTOM_EXTRACTION = "symptom_extraction"
    ORCHESTRATOR = "orchestrator"
    TREE_MANAGER = "tree_manager"
    USER_INTERACTION = "user_interaction"
    SYSTEM = "system"
    WEBSOCKET = "websocket"
    # Add more as needed for your system components
