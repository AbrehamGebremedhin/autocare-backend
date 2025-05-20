from .base import AgentBase
from typing import Any, Dict, List

class OrchestratorAgent(AgentBase):
    """
    An agent that orchestrates multiple sub-agents and delegates tasks.
    """
    def __init__(self, services: Dict[str, Any], config: Dict[str, Any] = None, sub_agents: List[AgentBase] = None):
        super().__init__(services, config)
        self.sub_agents = sub_agents or []

    async def can_handle(self, task: str, context: Dict[str, Any]) -> bool:
        # The orchestrator can handle any task if at least one sub-agent can handle it
        for agent in self.sub_agents:
            if await agent.can_handle(task, context):
                return True
        return False

    async def handle(self, task: str, context: Dict[str, Any]) -> Any:
        # Delegate the task to the first capable sub-agent
        for agent in self.sub_agents:
            if await agent.can_handle(task, context):
                result = await agent.handle(task, context)
                return await self.post_process(result, context)
        raise NotImplementedError("No sub-agent can handle the given task.")

    async def post_process(self, result: Any, context: Dict[str, Any]) -> Any:
        # Optionally override for orchestrator-specific post-processing
        return await super().post_process(result, context)
