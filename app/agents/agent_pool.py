"""
Enhanced Agent Pool Manager for handling multiple concurrent agent instances.
Provides agent lifecycle management, resource pooling, and load balancing.
"""
import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type, Set
from enum import Enum
from contextlib import asynccontextmanager

from app.agents.base_agent import BaseAgent, AgentState
from app.agents.orchestrator_agent import OrchestratorAgent
from app.agents.symptom_extraction_agent import SymptomExtractorAgent
from app.agents.diagnostic_agent import DiagnosisAgent
from app.agents.user_interaction_agent import UserInteractionAgent
from app.utils.logger import get_logger_instance
from app.utils.session_manager import get_session_manager, ConcurrentSessionManager
from app.utils.exceptions import AgentException, ResourceException


class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    SYMPTOM_EXTRACTOR = "symptom_extractor"
    DIAGNOSTIC = "diagnostic"
    USER_INTERACTION = "user_interaction"


@dataclass
class AgentInstance:
    """Tracks individual agent instances in the pool"""
    instance_id: str
    agent_type: AgentType
    agent: BaseAgent
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    is_busy: bool = False
    current_session_id: Optional[str] = None
    current_user_id: Optional[str] = None
    error_count: int = 0
    
    def mark_busy(self, session_id: str = None, user_id: str = None):
        """Mark agent as busy"""
        self.is_busy = True
        self.current_session_id = session_id
        self.current_user_id = user_id
        self.last_used = datetime.utcnow()
        self.use_count += 1
    
    def mark_idle(self):
        """Mark agent as idle"""
        self.is_busy = False
        self.current_session_id = None
        self.current_user_id = None
    
    def mark_error(self):
        """Mark agent as having an error"""
        self.error_count += 1
        self.mark_idle()
    
    def is_stale(self, max_age_hours: int = 2) -> bool:
        """Check if agent instance is stale"""
        age = datetime.utcnow() - self.last_used
        return age > timedelta(hours=max_age_hours)
    
    def is_overused(self, max_uses: int = 100) -> bool:
        """Check if agent has been used too many times"""
        return self.use_count > max_uses


class ConcurrentAgentPool:
    """
    Manages pools of agents for high concurrency scenarios.
    Provides agent lifecycle management, load balancing, and resource optimization.
    """
    
    def __init__(self, session_manager: Optional[ConcurrentSessionManager] = None):
        self.logger = get_logger_instance("AgentPool")
        self.session_manager = session_manager
        
        # Agent pools by type
        self._agent_pools: Dict[AgentType, List[AgentInstance]] = {
            agent_type: [] for agent_type in AgentType
        }
        
        # Agent classes for instantiation
        self._agent_classes = {
            AgentType.ORCHESTRATOR: OrchestratorAgent,
            AgentType.SYMPTOM_EXTRACTOR: SymptomExtractorAgent,
            AgentType.DIAGNOSTIC: DiagnosisAgent,
            AgentType.USER_INTERACTION: UserInteractionAgent
        }
        
        # Pool configuration
        self._pool_config = {
            AgentType.ORCHESTRATOR: {'min': 2, 'max': 10, 'max_uses': 50},
            AgentType.SYMPTOM_EXTRACTOR: {'min': 3, 'max': 15, 'max_uses': 100},
            AgentType.DIAGNOSTIC: {'min': 3, 'max': 15, 'max_uses': 100},
            AgentType.USER_INTERACTION: {'min': 2, 'max': 10, 'max_uses': 200}
        }
        
        # Thread safety
        self._pool_locks: Dict[AgentType, asyncio.Lock] = {
            agent_type: asyncio.Lock() for agent_type in AgentType
        }
        self._global_lock = asyncio.Lock()
        
        # Performance tracking
        self._stats = {
            'total_agents_created': 0,
            'total_agents_destroyed': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'concurrent_peak': 0,
            'agent_errors': 0
        }
        
        # Background maintenance
        self._maintenance_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize the agent pool"""
        try:
            if self.session_manager is None:
                self.session_manager = await get_session_manager()
            
            # Create initial agent instances
            await self._create_initial_pools()
            
            # Start maintenance task
            self._maintenance_task = asyncio.create_task(self._pool_maintenance())
            
            await self.logger.info("Agent pool initialized successfully")
        except Exception as e:
            await self.logger.error(f"Failed to initialize agent pool: {str(e)}")
            raise AgentException(f"Pool initialization failed: {str(e)}")
    
    async def _create_initial_pools(self):
        """Create initial instances for each agent type"""
        for agent_type in AgentType:
            # Skip creating session-specific agents during initialization
            if agent_type in [AgentType.SYMPTOM_EXTRACTOR, AgentType.DIAGNOSTIC]:
                await self.logger.info(f"Skipping pre-creation of {agent_type.value} agents (session-specific)")
                continue
                
            config = self._pool_config[agent_type]
            min_instances = config['min']
            
            tasks = []
            for _ in range(min_instances):
                task = asyncio.create_task(self._create_agent_instance(agent_type))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful = 0
            for result in results:
                if isinstance(result, AgentInstance):
                    successful += 1
                else:
                    await self.logger.error(f"Failed to create {agent_type.value} agent: {result}")
            
            await self.logger.info(f"Created {successful}/{min_instances} {agent_type.value} agents")
    
    async def _create_agent_instance(self, agent_type: AgentType, **kwargs) -> AgentInstance:
        """Create a new agent instance"""
        try:
            instance_id = f"{agent_type.value}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Get agent class and create instance
            agent_class = self._agent_classes[agent_type]
            
            # Create agent with proper dependencies
            # Only create agents that don't require session-specific parameters during pool init
            if agent_type == AgentType.ORCHESTRATOR:
                agent = agent_class(
                    websocket_manager=kwargs.get('websocket_manager'),
                    **kwargs
                )
                # Initialize the agent
                await agent.initialize()
            elif agent_type == AgentType.USER_INTERACTION:
                agent = agent_class(**kwargs)
                # Initialize the agent
                await agent.initialize()
            elif agent_type in [AgentType.SYMPTOM_EXTRACTOR, AgentType.DIAGNOSTIC]:
                # These agents require session-specific parameters (car_id, diagnosis_tree)
                # We'll create them on-demand, so just store the class for now
                agent = None  # Will be created when requested
            else:
                agent = agent_class(**kwargs)
                # Initialize the agent
                await agent.initialize()
            
            # Create instance wrapper
            agent_instance = AgentInstance(
                instance_id=instance_id,
                agent_type=agent_type,
                agent=agent,  # May be None for session-specific agents
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow()
            )
            
            # Only add to pool if agent was successfully created
            if agent is not None:
                async with self._pool_locks[agent_type]:
                    self._agent_pools[agent_type].append(agent_instance)
            
            self._stats['total_agents_created'] += 1
            
            await self.logger.debug(f"Created agent instance: {instance_id}")
            return agent_instance
            
        except Exception as e:
            await self.logger.error(f"Failed to create {agent_type.value} agent: {str(e)}")
            raise AgentException(f"Agent creation failed: {str(e)}")
    
    @asynccontextmanager
    async def get_agent(
        self, 
        agent_type: AgentType, 
        session_id: str = None, 
        user_id: str = None,
        **agent_kwargs
    ):
        """
        Get an agent from the pool with automatic cleanup.
        """
        agent_instance = None
        
        try:
            # Get agent from pool
            agent_instance = await self._acquire_agent(agent_type, session_id, user_id, **agent_kwargs)
            
            if not agent_instance:
                raise ResourceException(f"No available {agent_type.value} agents")
            
            # Track usage
            self._stats['pool_hits'] += 1
            
            # Update concurrent peak
            busy_count = sum(
                sum(1 for instance in pool if instance.is_busy)
                for pool in self._agent_pools.values()
            )
            if busy_count > self._stats['concurrent_peak']:
                self._stats['concurrent_peak'] = busy_count
            
            yield agent_instance.agent
            
        except Exception as e:
            await self.logger.error(f"Error using {agent_type.value} agent: {str(e)}")
            self._stats['agent_errors'] += 1
            
            if agent_instance:
                agent_instance.mark_error()
            
            raise
        finally:
            # Return agent to pool
            if agent_instance:
                await self._release_agent(agent_instance)
    
    async def _acquire_agent(
        self, 
        agent_type: AgentType, 
        session_id: str = None, 
        user_id: str = None,
        **agent_kwargs
    ) -> Optional[AgentInstance]:
        """Acquire an agent from the pool"""
        async with self._pool_locks[agent_type]:
            pool = self._agent_pools[agent_type]
            
            # Find idle agent
            for agent_instance in pool:
                if not agent_instance.is_busy and not agent_instance.is_stale():
                    agent_instance.mark_busy(session_id, user_id)
                    
                    # Configure agent if needed
                    await self._configure_agent(agent_instance, **agent_kwargs)
                    
                    await self.logger.debug(f"Acquired agent: {agent_instance.instance_id}")
                    return agent_instance
            
            # No idle agents available, create new one if under limit
            config = self._pool_config[agent_type]
            if len(pool) < config['max']:
                try:
                    agent_instance = await self._create_agent_instance(agent_type, **agent_kwargs)
                    agent_instance.mark_busy(session_id, user_id)
                    
                    await self.logger.debug(f"Created new agent: {agent_instance.instance_id}")
                    return agent_instance
                except Exception as e:
                    await self.logger.error(f"Failed to create new {agent_type.value} agent: {str(e)}")
            
            # No agents available
            self._stats['pool_misses'] += 1
            return None
    
    async def _configure_agent(self, agent_instance: AgentInstance, **kwargs):
        """Configure agent instance with specific parameters"""
        try:
            agent = agent_instance.agent
            
            # Configure car-specific agents
            if agent_instance.agent_type in [AgentType.SYMPTOM_EXTRACTOR, AgentType.DIAGNOSTIC]:
                car_id = kwargs.get('car_id')
                if car_id and hasattr(agent, 'car_id'):
                    agent.car_id = car_id
                
                diagnosis_tree = kwargs.get('diagnosis_tree')
                if diagnosis_tree and hasattr(agent, 'diagnosis_tree'):
                    agent.diagnosis_tree = diagnosis_tree
            
            # Configure other agent-specific parameters
            websocket = kwargs.get('websocket')
            if websocket and hasattr(agent, 'websocket_manager'):
                # Update websocket manager if needed
                pass
            
        except Exception as e:
            await self.logger.warning(f"Failed to configure agent {agent_instance.instance_id}: {str(e)}")
    
    async def _release_agent(self, agent_instance: AgentInstance):
        """Release agent back to the pool"""
        try:
            agent_instance.mark_idle()
            
            # Check if agent should be retired
            config = self._pool_config[agent_instance.agent_type]
            if (agent_instance.is_overused(config['max_uses']) or 
                agent_instance.is_stale() or 
                agent_instance.error_count > 5):
                
                await self._destroy_agent_instance(agent_instance)
            
            await self.logger.debug(f"Released agent: {agent_instance.instance_id}")
            
        except Exception as e:
            await self.logger.error(f"Error releasing agent {agent_instance.instance_id}: {str(e)}")
    
    async def _destroy_agent_instance(self, agent_instance: AgentInstance):
        """Destroy an agent instance and remove from pool"""
        try:
            async with self._pool_locks[agent_instance.agent_type]:
                pool = self._agent_pools[agent_instance.agent_type]
                if agent_instance in pool:
                    pool.remove(agent_instance)
            
            # Cleanup agent
            if hasattr(agent_instance.agent, 'close'):
                if asyncio.iscoroutinefunction(agent_instance.agent.close):
                    await agent_instance.agent.close()
                else:
                    agent_instance.agent.close()
            
            self._stats['total_agents_destroyed'] += 1
            
            await self.logger.debug(f"Destroyed agent: {agent_instance.instance_id}")
            
        except Exception as e:
            await self.logger.error(f"Error destroying agent {agent_instance.instance_id}: {str(e)}")
    
    async def _pool_maintenance(self):
        """Background task for pool maintenance"""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_stale_agents()
                await self._ensure_minimum_agents()
                await asyncio.sleep(60)  # Run every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.logger.error(f"Error in pool maintenance: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup_stale_agents(self):
        """Remove stale agents from pools"""
        for agent_type in AgentType:
            async with self._pool_locks[agent_type]:
                pool = self._agent_pools[agent_type]
                stale_agents = [
                    agent for agent in pool
                    if not agent.is_busy and (agent.is_stale() or agent.error_count > 5)
                ]
                
                for agent in stale_agents:
                    await self._destroy_agent_instance(agent)
                
                if stale_agents:
                    await self.logger.info(f"Cleaned up {len(stale_agents)} stale {agent_type.value} agents")
    
    async def _ensure_minimum_agents(self):
        """Ensure minimum number of agents in each pool"""
        for agent_type in AgentType:
            config = self._pool_config[agent_type]
            min_agents = config['min']
            
            async with self._pool_locks[agent_type]:
                pool = self._agent_pools[agent_type]
                idle_count = sum(1 for agent in pool if not agent.is_busy)
                
                if idle_count < min_agents:
                    needed = min_agents - idle_count
                    
                    # Create needed agents
                    tasks = []
                    for _ in range(needed):
                        task = asyncio.create_task(self._create_agent_instance(agent_type))
                        tasks.append(task)
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        successful = sum(1 for result in results if isinstance(result, AgentInstance))
                        
                        if successful > 0:
                            await self.logger.info(f"Created {successful} additional {agent_type.value} agents")
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        pool_stats = {}
        
        for agent_type in AgentType:
            async with self._pool_locks[agent_type]:
                pool = self._agent_pools[agent_type]
                
                total = len(pool)
                busy = sum(1 for agent in pool if agent.is_busy)
                idle = total - busy
                errors = sum(agent.error_count for agent in pool)
                
                pool_stats[agent_type.value] = {
                    'total': total,
                    'busy': busy,
                    'idle': idle,
                    'error_count': errors,
                    'utilization': (busy / total * 100) if total > 0 else 0
                }
        
        return {
            **self._stats,
            'pool_details': pool_stats,
            'total_agents': sum(len(pool) for pool in self._agent_pools.values())
        }
    
    async def shutdown(self):
        """Shutdown the agent pool"""
        try:
            self._shutdown_event.set()
            
            # Cancel maintenance task
            if self._maintenance_task:
                self._maintenance_task.cancel()
                try:
                    await self._maintenance_task
                except asyncio.CancelledError:
                    pass
            
            # Destroy all agents
            for agent_type in AgentType:
                async with self._pool_locks[agent_type]:
                    pool = self._agent_pools[agent_type].copy()
                    self._agent_pools[agent_type].clear()
                
                for agent_instance in pool:
                    await self._destroy_agent_instance(agent_instance)
            
            await self.logger.info("Agent pool shutdown complete")
            
        except Exception as e:
            await self.logger.error(f"Error during agent pool shutdown: {str(e)}")


# Global agent pool instance
agent_pool = ConcurrentAgentPool()


async def get_agent_pool() -> ConcurrentAgentPool:
    """Get the agent pool instance"""
    if not hasattr(agent_pool, '_initialized'):
        await agent_pool.initialize()
        agent_pool._initialized = True
    return agent_pool


# Convenience functions for getting specific agent types
async def get_orchestrator_agent(**kwargs):
    """Get an orchestrator agent from the pool"""
    pool = await get_agent_pool()
    return pool.get_agent(AgentType.ORCHESTRATOR, **kwargs)


async def get_symptom_extractor_agent(**kwargs):
    """Get a symptom extractor agent from the pool"""
    pool = await get_agent_pool()
    return pool.get_agent(AgentType.SYMPTOM_EXTRACTOR, **kwargs)


async def get_diagnostic_agent(**kwargs):
    """Get a diagnostic agent from the pool"""
    pool = await get_agent_pool()
    return pool.get_agent(AgentType.DIAGNOSTIC, **kwargs)


async def get_user_interaction_agent(**kwargs):
    """Get a user interaction agent from the pool"""
    pool = await get_agent_pool()
    return pool.get_agent(AgentType.USER_INTERACTION, **kwargs)
