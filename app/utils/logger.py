import logging
import asyncio
import functools
from typing import Optional
from app.core.interfaces import ILogger

class Logger(ILogger):
    _instances = {}
    _lock = asyncio.Lock()

    def __new__(cls, name: str = "autocare"):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]

    def __init__(self, name: str = "autocare"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(log_type)s] - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self._initialized = True

    async def info(self, message: str):
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(
                None,
                functools.partial(self.logger.info, message, extra={"log_type": "info"})
            )

    async def warning(self, message: str):
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(
                None,
                functools.partial(self.logger.warning, message, extra={"log_type": "warning"})
            )

    async def error(self, message: str):
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(
                None,
                functools.partial(self.logger.error, message, extra={"log_type": "error"})
            )

    async def debug(self, message: str):
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(
                None,
                functools.partial(self.logger.debug, message, extra={"log_type": "debug"})
            )

    async def critical(self, message: str):
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(
                None,
                functools.partial(self.logger.critical, message, extra={"log_type": "critical"})
            )

def get_logger_instance(name: Optional[str] = None) -> Logger:
    return Logger(name or "autocare")
