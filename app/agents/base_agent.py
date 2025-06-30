from typing import Any, Optional
import re
from app.utils.logger import get_logger_instance
from app.utils.websocket import manager
from app.core.interfaces import ILogger, IWebSocketManager
from app.utils.message_types import MessageType, MessageSource
import abc

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
        Base class for all agents. Handles car info, logging, and websocket communication.
        """
        self.car_crud = car_crud
        self.car_id = car_id
        self.car_make = car_make
        self.car_model = car_model
        self.car_year = car_year
        self.logger = logger or get_logger_instance(logger_name or self.__class__.__name__)
        self.websocket_manager = websocket_manager or manager

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
