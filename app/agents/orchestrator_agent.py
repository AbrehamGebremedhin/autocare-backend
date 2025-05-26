import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from .base import AgentBase

class OrchestratorAgent(AgentBase):
    """
    Enhanced orchestrator agent that orchestrates multiple sub-agents with:
    - Parallel agent capability checking
    - Load balancing across agents
    - Task prioritization and routing
    - Performance monitoring
    - Fallback handling and error recovery
    """
    
    def __init__(self, services: Dict[str, Any], config: Dict[str, Any] = None, sub_agents: List[AgentBase] = None):
        super().__init__(services, config)
        self.sub_agents = sub_agents or []
        self.agent_loads: Dict[str, int] = {agent.__class__.__name__: 0 for agent in self.sub_agents}
        self.agent_performance: Dict[str, List[float]] = {agent.__class__.__name__: [] for agent in self.sub_agents}
        self.max_parallel_checks = config.get('max_parallel_checks', 5) if config else 5
        self.enable_load_balancing = config.get('enable_load_balancing', True) if config else True
        self.task_priority_map = config.get('task_priority_map', {}) if config else {}
        
    async def can_handle(self, task: str, context: Dict[str, Any]) -> bool:
        """Check if any sub-agent can handle the task using parallel processing."""
        cache_key = f"can_handle:{task}:{hash(frozenset(context.items()) if context else 0)}"
        cached = await self.get_cached_result(cache_key)
        if cached is not None:
            return cached
            
        if not self.sub_agents:
            result = False
        else:
            # Use parallel checking for faster response
            results = await self._parallel_capability_check(task, context)
            result = any(results)
            
        await self.cache_result(cache_key, result, ttl=300)  # 5 min cache
        return result
    
    async def _parallel_capability_check(self, task: str, context: Dict[str, Any]) -> List[bool]:
        """Check agent capabilities in parallel with controlled concurrency."""
        semaphore = asyncio.Semaphore(self.max_parallel_checks)
        
        async def check_agent(agent: AgentBase) -> bool:
            async with semaphore:
                try:
                    return await agent.can_handle(task, context)
                except Exception as e:
                    await self.logger.error(f"Error checking {agent.__class__.__name__} capability: {e}")
                    return False
        
        tasks = [check_agent(agent) for agent in self.sub_agents]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def handle(self, task: str, context: Dict[str, Any]) -> Any:
        """Handle task with intelligent agent selection and error recovery."""
        start_time = time.time()
        
        try:
            # Find capable agents
            capable_agents = await self._find_capable_agents(task, context)
            if not capable_agents:
                raise NotImplementedError("No sub-agent can handle the given task.")
            
            # Select best agent based on load balancing and performance
            selected_agent = await self._select_best_agent(capable_agents, task, context)
            
            # Update load tracking
            agent_name = selected_agent.__class__.__name__
            self.agent_loads[agent_name] = self.agent_loads.get(agent_name, 0) + 1
            
            try:
                # Execute task with the selected agent
                result = await selected_agent.handle(task, context)
                
                # Track performance
                execution_time = time.time() - start_time
                self._update_performance_metrics(agent_name, execution_time)
                
                # Post-process result
                return await self.post_process(result, context)
                
            finally:
                # Decrease load
                self.agent_loads[agent_name] = max(0, self.agent_loads[agent_name] - 1)
                
        except Exception as e:
            # Try fallback handling
            return await self._handle_fallback(task, context, e)
    
    async def _find_capable_agents(self, task: str, context: Dict[str, Any]) -> List[AgentBase]:
        """Find all agents capable of handling the task."""
        capabilities = await self._parallel_capability_check(task, context)
        return [agent for agent, capable in zip(self.sub_agents, capabilities) if capable]
    
    async def _select_best_agent(self, capable_agents: List[AgentBase], task: str, context: Dict[str, Any]) -> AgentBase:
        """Select the best agent based on load balancing and performance."""
        if len(capable_agents) == 1:
            return capable_agents[0]
        
        if not self.enable_load_balancing:
            return capable_agents[0]
        
        # Consider task priority
        priority = self.task_priority_map.get(task, 0)
        
        # Score agents based on load and performance
        best_agent = None
        best_score = float('inf')
        
        for agent in capable_agents:
            agent_name = agent.__class__.__name__
            current_load = self.agent_loads.get(agent_name, 0)
            avg_performance = self._get_average_performance(agent_name)
            
            # Lower score is better (load + performance time)
            score = current_load + avg_performance - priority
            
            if score < best_score:
                best_score = score
                best_agent = agent
        
        return best_agent or capable_agents[0]
    
    def _get_average_performance(self, agent_name: str) -> float:
        """Get average performance time for an agent."""
        performances = self.agent_performance.get(agent_name, [])
        if not performances:
            return 0.0
        return sum(performances[-10:]) / len(performances[-10:])  # Last 10 measurements
    
    def _update_performance_metrics(self, agent_name: str, execution_time: float):
        """Update performance metrics for an agent."""
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = []
        
        self.agent_performance[agent_name].append(execution_time)
        
        # Keep only last 50 measurements
        if len(self.agent_performance[agent_name]) > 50:
            self.agent_performance[agent_name] = self.agent_performance[agent_name][-50:]
    
    async def _handle_fallback(self, task: str, context: Dict[str, Any], original_error: Exception) -> Any:
        """Handle fallback when primary execution fails."""
        await self.logger.warning(f"Primary execution failed: {original_error}. Attempting fallback.")
        
        # Try with simplified context
        simplified_context = {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool))}
        
        try:
            capable_agents = await self._find_capable_agents(task, simplified_context)
            if capable_agents:
                # Try with the agent that has best historical performance
                best_performing_agent = min(
                    capable_agents,
                    key=lambda a: self._get_average_performance(a.__class__.__name__)
                )
                result = await best_performing_agent.handle(task, simplified_context)
                return await self.post_process(result, simplified_context)
        except Exception as fallback_error:
            await self.logger.error(f"Fallback also failed: {fallback_error}")
        
        # If all else fails, raise the original error
        raise original_error
    
    async def post_process(self, result: Any, context: Dict[str, Any]) -> Any:
        """Enhanced post-processing with orchestrator-specific logic."""
        # Add orchestration metadata
        orchestration_metadata = {
            'orchestrator': self.__class__.__name__,
            'agent_loads': self.agent_loads.copy(),
            'timestamp': time.time()
        }
        
        # If result is a dict, add metadata
        if isinstance(result, dict):
            result['_orchestration_metadata'] = orchestration_metadata
        
        return await super().post_process(result, context)
    
    async def get_agent_statistics(self) -> Dict[str, Any]:
        """Get current agent statistics for monitoring."""
        return {
            'agent_loads': self.agent_loads.copy(),
            'agent_performance': {
                name: {
                    'average': self._get_average_performance(name),
                    'count': len(times),
                    'recent': times[-5:] if times else []
                }
                for name, times in self.agent_performance.items()
            },
            'total_agents': len(self.sub_agents),
            'active_agents': sum(1 for load in self.agent_loads.values() if load > 0)
        }
    
    async def add_agent(self, agent: AgentBase):
        """Dynamically add a new sub-agent."""
        if agent not in self.sub_agents:
            self.sub_agents.append(agent)
            agent_name = agent.__class__.__name__
            self.agent_loads[agent_name] = 0
            self.agent_performance[agent_name] = []
            await self.logger.info(f"Added new agent: {agent_name}")
    
    async def remove_agent(self, agent: AgentBase):
        """Dynamically remove a sub-agent."""
        if agent in self.sub_agents:
            self.sub_agents.remove(agent)
            agent_name = agent.__class__.__name__
            self.agent_loads.pop(agent_name, None)
            self.agent_performance.pop(agent_name, None)
            await self.logger.info(f"Removed agent: {agent_name}")
