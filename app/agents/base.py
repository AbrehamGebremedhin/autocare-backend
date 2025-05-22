from abc import ABC, abstractmethod
from typing import Any, Dict

class AgentBase(ABC):
    """
    Abstract base class for all agents.
    """
    @abstractmethod
    async def pre_process(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optional pre-processing before handling a task.
        """
        return context

    @abstractmethod
    async def handle(self, task: str, context: Dict[str, Any]) -> Any:
        """
        Handles the given task and returns the result.
        """
        pass

    async def post_process(self, result: Any, context: Dict[str, Any]) -> Any:
        """
        Optional post-processing after handling a task.
        """
        return result
