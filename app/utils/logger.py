import logging
from typing import Optional

class Logger:
    _instances = {}

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

    # Async so ~150 call sites keep awaiting them, but each is one direct call: a log line costs microseconds,
    # and a thread hop would queue it behind Whisper in the default executor.
    async def info(self, message: str):
        self.logger.info(message, extra={"log_type": "info"})

    async def warning(self, message: str):
        self.logger.warning(message, extra={"log_type": "warning"})

    async def error(self, message: str):
        self.logger.error(message, extra={"log_type": "error"})

    async def debug(self, message: str):
        self.logger.debug(message, extra={"log_type": "debug"})

    async def critical(self, message: str):
        self.logger.critical(message, extra={"log_type": "critical"})

def get_logger_instance(name: Optional[str] = None) -> Logger:
    return Logger(name or "autocare")
