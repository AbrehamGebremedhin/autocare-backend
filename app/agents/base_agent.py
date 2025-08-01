from typing import Any, Optional, Dict
import re
import asyncio
from enum import Enum
from app.utils.logger import get_logger_instance
from app.utils.websocket import manager
from app.core.interfaces import ILogger, IWebSocketManager
from app.utils.message_types import MessageType, MessageSource
import abc

class AgentState(Enum):
    """Agent lifecycle states"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class AgentCommand(abc.ABC):
    """Base command interface for agent operations"""
    
    @abc.abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the command with given context"""
        pass
    
    @abc.abstractmethod
    def validate(self, context: Dict[str, Any]) -> bool:
        """Validate the command can be executed with given context"""
        pass

class BaseAgent(abc.ABC):
    def __init__(
        self,
        car_crud: Optional[Any] = None,
        car_id: Optional[str] = None,
        car_make: Optional[str] = None,
        car_model: Optional[str] = None,
        car_year: Optional[str] = None,
        logger_name: Optional[str] = None,
        websocket_manager: Optional[IWebSocketManager] = None,
        logger: Optional[ILogger] = None
    ):
        """
        Base class for all agents. Handles car info, logging, websocket communication, and lifecycle management.
        """
        self.car_crud = car_crud
        self.car_id = car_id
        self.car_make = car_make
        self.car_model = car_model
        self.car_year = car_year
        self.logger = logger or get_logger_instance(logger_name or self.__class__.__name__)
        self.websocket_manager = websocket_manager or manager
        
        # Agent lifecycle management
        self._state = AgentState.INACTIVE
        self._commands: Dict[str, AgentCommand] = {}
        self._processing_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        
    @property
    def state(self) -> AgentState:
        """Get current agent state"""
        return self._state
    
    async def _set_state(self, new_state: AgentState) -> None:
        """Set agent state and log transition"""
        old_state = self._state
        self._state = new_state
        await self.logger.info(f"{self.__class__.__name__} state transition: {old_state.value} -> {new_state.value}")
    
    def register_command(self, name: str, command: AgentCommand) -> None:
        """Register a command for this agent"""
        self._commands[name] = command
    
    async def execute_command(self, command_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered command"""
        if command_name not in self._commands:
            raise ValueError(f"Command '{command_name}' not registered for {self.__class__.__name__}")
        
        command = self._commands[command_name]
        
        if not command.validate(context):
            raise ValueError(f"Command '{command_name}' validation failed")
        
        async with self._processing_lock:
            await self._set_state(AgentState.PROCESSING)
            try:
                result = await command.execute(context)
                await self._set_state(AgentState.ACTIVE)
                return result
            except Exception as e:
                await self._set_state(AgentState.ERROR)
                await self.logger.error(f"Command '{command_name}' execution failed: {str(e)}")
                raise
    
    async def initialize(self) -> None:
        """Initialize the agent"""
        await self._set_state(AgentState.INITIALIZING)
        await self._ensure_car_info()
        await self._set_state(AgentState.ACTIVE)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent"""
        await self._set_state(AgentState.SHUTDOWN)
        self._shutdown_event.set()
        await self.close()
    
    def is_healthy(self) -> bool:
        """Check if agent is in a healthy state"""
        return self._state not in [AgentState.ERROR, AgentState.SHUTDOWN]

    async def _ensure_car_info(self) -> None:
        """
        Ensure car_make, car_model, and car_year are set by fetching from DB if missing.
        """
        if not (self.car_make and self.car_model and self.car_year):
            if self.car_crud and self.car_id:
                car = await self.car_crud.get_car_by_id(self.car_id)
                if car:
                    self.car_make = self.car_make or car.get('make')
                    self.car_model = self.car_model or car.get('model')
                    self.car_year = self.car_year or car.get('year')

    def _sanitize_output(self, text: str) -> str:
        """
        Remove or rewrite any LLM output that tells the user to refer to the owner's manual or asks for car make/model/year.
        """
        text = re.sub(r"(?i)refer to (the|your) owner's manual[\.,]?", "", text)
        text = re.sub(r"(?i)see (the|your) owner's manual[\.,]?", "", text)
        text = re.sub(r"(?i)consult (the|your) owner's manual[\.,]?", "", text)
        text = re.sub(r"(?i)what is (the )?car'?s? (make|model|year)[\?\.]?", "", text)
        text = re.sub(r"(?i)please provide (the )?car'?s? (make|model|year)[\?\.]?", "", text)
        text = re.sub(r"(?i)can you tell me (the )?car'?s? (make|model|year)[\?\.]?", "", text)
        return text

    async def send_ws_message(self, websocket: Any, method: str, content: str, source: MessageSource, session_id: Optional[str] = None, details: Optional[Any] = None) -> None:
        """
        Generic method to send a websocket message using the specified method name.
        """
        if websocket is not None:
            from app.utils.json_utils import serialize_datetimes
            details = serialize_datetimes(details)
            ws_method = getattr(self.websocket_manager, method, None)
            if ws_method:
                await ws_method(websocket, content, source, session_id, details)

    async def broadcast_stage(self, stage: str) -> None:
        await self.websocket_manager.broadcast(stage)

    async def send_ws_info(self, websocket: Any, content: str, source: MessageSource, session_id: Optional[str] = None, details: Optional[Any] = None) -> None:
        await self.send_ws_message(websocket, 'send_info', content, source, session_id, details)

    async def send_ws_error(self, websocket: Any, content: str, source: MessageSource, session_id: Optional[str] = None, details: Optional[Any] = None) -> None:
        await self.send_ws_message(websocket, 'send_error', content, source, session_id, details)

    async def send_ws_progress(self, websocket: Any, content: str, source: MessageSource, progress: float, session_id: Optional[str] = None, details: Optional[Any] = None) -> None:
        if websocket is not None:
            from app.utils.json_utils import serialize_datetimes
            details = serialize_datetimes(details)
            await self.websocket_manager.send_progress(websocket, content, source, progress, session_id, details)

    async def send_ws_stage(self, websocket: Any, content: str, source: MessageSource, session_id: Optional[str] = None, details: Optional[Any] = None) -> None:
        await self.send_ws_message(websocket, 'send_stage', content, source, session_id, details)

    async def send_ws_result(self, websocket: Any, content: str, source: MessageSource, session_id: Optional[str] = None, details: Optional[Any] = None) -> None:
        await self.send_ws_message(websocket, 'send_result', content, source, session_id, details)

    async def send_ws_debug(self, websocket: Any, content: str, source: MessageSource, session_id: Optional[str] = None, details: Optional[Any] = None) -> None:
        await self.send_ws_message(websocket, 'send_debug', content, source, session_id, details)

    @abc.abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """
        Abstract method to be implemented by all agents for their main processing logic.
        """
        pass

    def close(self) -> None:
        """
        Optional cleanup method for agents to override if needed.
        """
        pass
