from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio
import time
from app.utils.logger import Logger
from app.services.base_service import BaseService

class AgentBase(ABC):
    """
    Enhanced abstract base class for all agents with performance optimizations.
    Integrates with the refactored service layer for improved caching, rate limiting, and monitoring.
    """
    
    def __init__(self, services: Dict[str, BaseService] = None, config: Dict[str, Any] = None, logger: Optional[Logger] = None):
        """
        Initialize the agent with enhanced services and configuration.
        Args:
            services: Dictionary of enhanced services (embedding, search, chat, etc.)
            config: Agent-specific configuration
            logger: Logger instance for monitoring
        """
        self.services = services or {}
        self.config = config or {}
        self.logger = logger or Logger(self.__class__.__name__)
        self._cache = {}
        self._cache_ttl = {}
        self._performance_metrics = {
            'task_count': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'error_count': 0
        }
        self.lock = asyncio.Lock()
    
    def _get_cache_key(self, task: str, context: Dict[str, Any]) -> str:
        """Generate cache key for task results."""
        context_str = str(sorted(context.items())) if context else ""
        return f"{self.__class__.__name__}:{task}:{hash(context_str)}"
    
    def _is_cache_valid(self, cache_key: str, ttl_seconds: int = 300) -> bool:
        """Check if cached result is still valid."""
        if cache_key not in self._cache:
            return False
        return time.time() < self._cache_ttl.get(cache_key, 0)
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve result from cache if valid."""
        async with self.lock:
            if self._is_cache_valid(cache_key):
                await self.logger.info(f"Cache hit for {cache_key}")
                return self._cache[cache_key]
            # Cleanup expired entries
            if cache_key in self._cache:
                del self._cache[cache_key]
                del self._cache_ttl[cache_key]
            return None
    
    async def _set_cache(self, cache_key: str, result: Any, ttl_seconds: int = 300) -> None:
        """Store result in cache with TTL."""
        async with self.lock:
            self._cache[cache_key] = result
            self._cache_ttl[cache_key] = time.time() + ttl_seconds
    
    async def _update_metrics(self, execution_time: float, success: bool = True):
        """Update performance metrics."""
        async with self.lock:
            self._performance_metrics['task_count'] += 1
            self._performance_metrics['total_time'] += execution_time
            self._performance_metrics['average_time'] = (
                self._performance_metrics['total_time'] / self._performance_metrics['task_count']
            )
            if not success:
                self._performance_metrics['error_count'] += 1
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        async with self.lock:
            return self._performance_metrics.copy()
    
    @abstractmethod
    async def can_handle(self, task: str, context: Dict[str, Any]) -> bool:
        """
        Determine if this agent can handle the given task.
        """
        pass
    
    @abstractmethod
    async def pre_process(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced pre-processing with logging and validation.
        """
        await self.logger.info(f"Pre-processing task: {task}")
        return context

    @abstractmethod
    async def handle(self, task: str, context: Dict[str, Any]) -> Any:
        """
        Enhanced task handling with caching, metrics, and error handling.
        """
        pass
    
    async def execute(self, task: str, context: Dict[str, Any], use_cache: bool = True, cache_ttl: int = 300) -> Any:
        """
        Execute task with enhanced caching, monitoring, and error handling.
        """
        start_time = time.time()
        cache_key = self._get_cache_key(task, context) if use_cache else None
        
        try:
            # Check cache first
            if use_cache and cache_key:
                cached_result = await self._get_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Pre-process
            processed_context = await self.pre_process(task, context)
            
            # Handle task
            result = await self.handle(task, processed_context)
            
            # Post-process
            final_result = await self.post_process(result, processed_context)
            
            # Cache result
            if use_cache and cache_key:
                await self._set_cache(cache_key, final_result, cache_ttl)
            
            # Update metrics
            execution_time = time.time() - start_time
            await self._update_metrics(execution_time, success=True)
            await self.logger.info(f"Task completed in {execution_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            await self._update_metrics(execution_time, success=False)
            await self.logger.error(f"Task failed after {execution_time:.2f}s: {e}")
            raise

    async def post_process(self, result: Any, context: Dict[str, Any]) -> Any:
        """
        Enhanced post-processing with logging and validation.
        """
        await self.logger.info(f"Post-processing completed")
        return result
    
    async def clear_cache(self) -> None:
        """Clear agent cache."""
        async with self.lock:
            self._cache.clear()
            self._cache_ttl.clear()
            await self.logger.info("Agent cache cleared")
