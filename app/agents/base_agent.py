from typing import Any
import re
from app.utils.logger import get_logger_instance
from app.utils.websocket import manager
from app.core.interfaces import ILogger, IWebSocketManager
from app.utils.message_types import MessageType, MessageSource

class BaseAgent:
    def __init__(self, car_crud=None, car_id=None, car_make=None, car_model=None, car_year=None, logger_name=None, websocket_manager: IWebSocketManager = None, logger: ILogger = None):
        self.car_crud = car_crud
        self.car_id = car_id
        self.car_make = car_make
        self.car_model = car_model
        self.car_year = car_year
        self.logger = logger or get_logger_instance(logger_name or self.__class__.__name__)
        self.websocket_manager = websocket_manager or manager

    async def _ensure_car_info(self):
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

    async def broadcast_stage(self, stage: str):
        await self.websocket_manager.broadcast(stage)

    async def send_ws_info(self, websocket, content, source: MessageSource, session_id=None, details=None):
        if websocket is not None:
            from app.utils.json_utils import serialize_datetimes
            details = serialize_datetimes(details)
            await self.websocket_manager.send_info(websocket, content, source, session_id, details)

    async def send_ws_error(self, websocket, content, source: MessageSource, session_id=None, details=None):
        if websocket is not None:
            from app.utils.json_utils import serialize_datetimes
            details = serialize_datetimes(details)
            await self.websocket_manager.send_error(websocket, content, source, session_id, details)

    async def send_ws_progress(self, websocket, content, source: MessageSource, progress: float, session_id=None, details=None):
        if websocket is not None:
            from app.utils.json_utils import serialize_datetimes
            details = serialize_datetimes(details)
            await self.websocket_manager.send_progress(websocket, content, source, progress, session_id, details)

    async def send_ws_stage(self, websocket, content, source: MessageSource, session_id=None, details=None):
        if websocket is not None:
            from app.utils.json_utils import serialize_datetimes
            details = serialize_datetimes(details)
            await self.websocket_manager.send_stage(websocket, content, source, session_id, details)

    async def send_ws_result(self, websocket, content, source: MessageSource, session_id=None, details=None):
        if websocket is not None:
            from app.utils.json_utils import serialize_datetimes
            details = serialize_datetimes(details)
            await self.websocket_manager.send_result(websocket, content, source, session_id, details)

    async def send_ws_debug(self, websocket, content, source: MessageSource, session_id=None, details=None):
        if websocket is not None:
            from app.utils.json_utils import serialize_datetimes
            details = serialize_datetimes(details)
            await self.websocket_manager.send_debug(websocket, content, source, session_id, details)
