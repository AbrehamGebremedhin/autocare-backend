import logging
import asyncio

class Logger:
    def __init__(self, name: str = "autocare"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(log_type)s] - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self._lock = asyncio.Lock()

    async def info(self, message: str):
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(None, self.logger.info, message, {"log_type": "info"})

    async def warning(self, message: str):
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(None, self.logger.warning, message, {"log_type": "warning"})

    async def error(self, message: str):
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(None, self.logger.error, message, {"log_type": "error"})

    async def debug(self, message: str):
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(None, self.logger.debug, message, {"log_type": "debug"})

    async def get_logger(self):
        return self.logger
