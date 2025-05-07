from abc import ABC, abstractmethod
from app.utils.websocket import manager

class BaseService(ABC):
    """
    Abstract base class for all service classes.
    """
    def __init__(self):
        self.websocket_manager = manager

    @abstractmethod
    def perform_action(self, *args, **kwargs):
        """
        Abstract method to be implemented by all services.
        """
        pass