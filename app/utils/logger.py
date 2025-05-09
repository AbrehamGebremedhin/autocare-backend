import logging

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

    def info(self, message: str):
        self.logger.info(message, extra={"log_type": "info"})

    def warning(self, message: str):
        self.logger.warning(message, extra={"log_type": "warning"})

    def error(self, message: str):
        self.logger.error(message, extra={"log_type": "error"})

    def debug(self, message: str):
        self.logger.debug(message, extra={"log_type": "debug"})

    def get_logger(self):
        return self.logger
