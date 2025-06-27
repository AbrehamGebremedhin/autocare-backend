from abc import ABC, abstractmethod
from app.utils.websocket import manager
from app.core.interfaces import IWebSocketManager
from typing import Optional, Dict, Any
import asyncio
import time
from functools import wraps

class BaseService(ABC):
    """
    Abstract base class for all service classes with performance enhancements.
    """
    def __init__(self, websocket_manager: IWebSocketManager = None):
        self.websocket_manager = websocket_manager or manager
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}
        self._default_cache_duration = 300  # 5 minutes default
        self._connection_pool = None
        self._rate_limit_tokens = 10
        self._rate_limit_interval = 1.0
        self._last_reset = time.time()

    def _get_cache_key(self, method_name: str, *args, **kwargs) -> str:
        """Generate a cache key for method calls."""
        key_parts = [method_name]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return "|".join(key_parts)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        ttl = self._cache_ttl.get(cache_key, 0)
        return time.time() < ttl

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from cache if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        # Cleanup expired cache entries
        if cache_key in self._cache:
            del self._cache[cache_key]
            del self._cache_ttl[cache_key]
        return None

    def _set_cache(self, cache_key: str, data: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store data in cache with TTL."""
        ttl = ttl_seconds or self._default_cache_duration
        self._cache[cache_key] = data
        self._cache_ttl[cache_key] = time.time() + ttl

    async def _rate_limit(self) -> None:
        """Simple token bucket rate limiting."""
        current_time = time.time()
        if current_time - self._last_reset >= self._rate_limit_interval:
            self._rate_limit_tokens = 10
            self._last_reset = current_time
        
        if self._rate_limit_tokens <= 0:
            sleep_time = self._rate_limit_interval - (current_time - self._last_reset)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self._rate_limit_tokens = 10
                self._last_reset = time.time()
        
        self._rate_limit_tokens -= 1

    @staticmethod
    def cache_result(ttl_seconds: Optional[int] = None):
        """Decorator to cache method results."""
        def decorator(func):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                cache_key = self._get_cache_key(func.__name__, *args, **kwargs)
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
                
                result = await func(self, *args, **kwargs)
                self._set_cache(cache_key, result, ttl_seconds)
                return result
            return wrapper
        return decorator

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._cache_ttl.clear()

    async def notify_task(self, event: str, detail: str = ""):
        """Send a websocket notification about task events (start/finish)."""
        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "service": self.__class__.__name__,
                "event": event,
                "detail": detail
            })

    async def run_with_notification(self, coro, *args, **kwargs):
        """Run a coroutine with start/finish notifications."""
        await self.notify_task("task_started")
        try:
            result = await coro(*args, **kwargs)
            await self.notify_task("task_finished")
            return result
        except Exception as exc:
            await self.notify_task("task_failed", detail=str(exc))
            raise

    @abstractmethod
    async def perform_action(self, *args, **kwargs):
        """
        Abstract async method to be implemented by all services.
        """
        pass