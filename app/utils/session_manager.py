"""
Advanced Session Manager for handling multiple concurrent users with proper isolation,
performance optimization, and horizontal scaling support.
"""
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from contextlib import asynccontextmanager

from app.utils.redis_cache import RedisCache, get_redis_cache
from app.utils.logger import get_logger_instance
from app.core.interfaces import IWebSocketManager
from app.utils.exceptions import SessionException, ValidationException


class SessionState(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


@dataclass
class UserSession:
    """Enhanced user session with isolation and metadata"""
    session_id: str
    user_id: str
    chat_session_id: Optional[str] = None
    state: SessionState = SessionState.ACTIVE
    created_at: datetime = None
    last_activity: datetime = None
    expires_at: datetime = None
    metadata: Dict[str, Any] = None
    websocket_connections: Set[str] = None
    active_agents: Set[str] = None
    resource_usage: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()
        if self.expires_at is None:
            self.expires_at = datetime.utcnow() + timedelta(hours=24)
        if self.metadata is None:
            self.metadata = {}
        if self.websocket_connections is None:
            self.websocket_connections = set()
        if self.active_agents is None:
            self.active_agents = set()
        if self.resource_usage is None:
            self.resource_usage = {
                'memory_mb': 0,
                'cpu_time_ms': 0,
                'db_queries': 0,
                'llm_tokens': 0,
                'websocket_messages': 0
            }
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at
    
    def extend_expiry(self, hours: int = 24):
        """Extend session expiry"""
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
    
    def add_websocket(self, connection_id: str):
        """Add WebSocket connection to session"""
        self.websocket_connections.add(connection_id)
        self.update_activity()
    
    def remove_websocket(self, connection_id: str):
        """Remove WebSocket connection from session"""
        self.websocket_connections.discard(connection_id)
    
    def add_agent(self, agent_name: str):
        """Track active agent for this session"""
        self.active_agents.add(agent_name)
    
    def remove_agent(self, agent_name: str):
        """Remove agent from active list"""
        self.active_agents.discard(agent_name)
    
    def update_resource_usage(self, **usage: Any):
        """Update resource usage metrics"""
        for key, value in usage.items():
            if key in self.resource_usage:
                self.resource_usage[key] += value
            else:
                self.resource_usage[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'chat_session_id': self.chat_session_id,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'metadata': self.metadata,
            'websocket_connections': list(self.websocket_connections),
            'active_agents': list(self.active_agents),
            'resource_usage': self.resource_usage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSession':
        """Create UserSession from dictionary"""
        return cls(
            session_id=data['session_id'],
            user_id=data['user_id'],
            chat_session_id=data.get('chat_session_id'),
            state=SessionState(data.get('state', 'active')),
            created_at=datetime.fromisoformat(data['created_at']),
            last_activity=datetime.fromisoformat(data['last_activity']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            metadata=data.get('metadata', {}),
            websocket_connections=set(data.get('websocket_connections', [])),
            active_agents=set(data.get('active_agents', [])),
            resource_usage=data.get('resource_usage', {})
        )


class ConcurrentSessionManager:
    """
    Advanced session manager optimized for high concurrency and horizontal scaling.
    Provides proper session isolation, resource tracking, and distributed state management.
    """
    
    def __init__(self, redis_cache: Optional[RedisCache] = None):
        self.redis_cache = redis_cache
        self.logger = get_logger_instance("SessionManager")
        
        # Local cache for frequently accessed sessions (with TTL)
        self._local_cache: Dict[str, UserSession] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._local_cache_ttl = 300  # 5 minutes
        
        # Session configuration
        self.max_sessions_per_user = 5
        self.session_timeout_hours = 24
        self.cleanup_interval = 3600  # 1 hour
        self.max_concurrent_sessions = 10000
        
        # Performance tracking
        self._stats = {
            'total_sessions_created': 0,
            'total_sessions_expired': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'concurrent_peak': 0
        }
        
        # Locks for thread safety
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize the session manager"""
        try:
            if self.redis_cache is None:
                self.redis_cache = await get_redis_cache()
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
            
            await self.logger.info("SessionManager initialized successfully")
        except Exception as e:
            await self.logger.error(f"Failed to initialize SessionManager: {str(e)}")
            raise SessionException(f"Initialization failed: {str(e)}")
    
    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session storage"""
        return f"session:{session_id}"
    
    def _get_user_sessions_key(self, user_id: str) -> str:
        """Get Redis key for user sessions list"""
        return f"user_sessions:{user_id}"
    
    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create lock for session-specific operations"""
        async with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = asyncio.Lock()
            return self._locks[session_id]
    
    async def _cleanup_local_cache(self):
        """Clean up expired entries from local cache"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > self._local_cache_ttl
        ]
        
        for key in expired_keys:
            self._local_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    async def create_session(
        self, 
        user_id: str, 
        chat_session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserSession:
        """
        Create a new user session with proper isolation and validation.
        """
        try:
            # Validate user_id
            if not user_id or len(user_id) > 100:
                raise ValidationException("Invalid user_id")
            
            # Check concurrent session limits
            user_sessions = await self.get_user_sessions(user_id)
            active_sessions = [s for s in user_sessions if s.state == SessionState.ACTIVE]
            
            if len(active_sessions) >= self.max_sessions_per_user:
                # Cleanup oldest session
                oldest_session = min(active_sessions, key=lambda s: s.last_activity)
                await self.terminate_session(oldest_session.session_id)
                await self.logger.warning(f"Terminated oldest session for user {user_id} due to limit")
            
            # Generate unique session ID
            session_id = f"{user_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Create session object
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                chat_session_id=chat_session_id,
                metadata=metadata or {}
            )
            
            # Store in Redis
            await self._store_session(session)
            
            # Add to user sessions list
            await self._add_to_user_sessions(user_id, session_id)
            
            # Update local cache
            self._local_cache[session_id] = session
            self._cache_timestamps[session_id] = time.time()
            
            # Update stats
            self._stats['total_sessions_created'] += 1
            
            await self.logger.info(f"Created session {session_id} for user {user_id}")
            return session
            
        except Exception as e:
            await self.logger.error(f"Failed to create session for user {user_id}: {str(e)}")
            raise SessionException(f"Session creation failed: {str(e)}")
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get session with local caching and automatic cleanup.
        """
        try:
            # Check local cache first
            if session_id in self._local_cache:
                cached_time = self._cache_timestamps.get(session_id, 0)
                if time.time() - cached_time < self._local_cache_ttl:
                    self._stats['cache_hits'] += 1
                    session = self._local_cache[session_id]
                    
                    # Check if session is expired
                    if session.is_expired():
                        await self.terminate_session(session_id)
                        return None
                    
                    return session
            
            # Cache miss - fetch from Redis
            self._stats['cache_misses'] += 1
            
            session_data = await self.redis_cache.get(self._get_session_key(session_id))
            if not session_data:
                return None
            
            session = UserSession.from_dict(session_data)
            
            # Check if session is expired
            if session.is_expired():
                await self.terminate_session(session_id)
                return None
            
            # Update local cache
            self._local_cache[session_id] = session
            self._cache_timestamps[session_id] = time.time()
            
            return session
            
        except Exception as e:
            await self.logger.error(f"Failed to get session {session_id}: {str(e)}")
            return None
    
    async def update_session(self, session: UserSession) -> bool:
        """
        Update session in both Redis and local cache.
        """
        try:
            lock = await self._get_lock(session.session_id)
            async with lock:
                session.update_activity()
                
                # Store in Redis
                await self._store_session(session)
                
                # Update local cache
                self._local_cache[session.session_id] = session
                self._cache_timestamps[session.session_id] = time.time()
                
                return True
                
        except Exception as e:
            await self.logger.error(f"Failed to update session {session.session_id}: {str(e)}")
            return False
    
    async def terminate_session(self, session_id: str) -> bool:
        """
        Terminate session and cleanup resources.
        """
        try:
            lock = await self._get_lock(session_id)
            async with lock:
                session = await self.get_session(session_id)
                if not session:
                    return True  # Already terminated
                
                # Update state
                session.state = SessionState.EXPIRED
                
                # Remove from user sessions list
                await self._remove_from_user_sessions(session.user_id, session_id)
                
                # Remove from Redis
                await self.redis_cache.delete(self._get_session_key(session_id))
                
                # Remove from local cache
                self._local_cache.pop(session_id, None)
                self._cache_timestamps.pop(session_id, None)
                
                # Cleanup lock
                self._locks.pop(session_id, None)
                
                self._stats['total_sessions_expired'] += 1
                
                await self.logger.info(f"Terminated session {session_id}")
                return True
                
        except Exception as e:
            await self.logger.error(f"Failed to terminate session {session_id}: {str(e)}")
            return False
    
    async def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """
        Get all active sessions for a user.
        """
        try:
            session_ids_data = await self.redis_cache.get(self._get_user_sessions_key(user_id))
            if not session_ids_data:
                return []
            
            session_ids = session_ids_data if isinstance(session_ids_data, list) else []
            sessions = []
            
            # Fetch each session
            for session_id in session_ids:
                session = await self.get_session(session_id)
                if session and session.state == SessionState.ACTIVE:
                    sessions.append(session)
            
            return sessions
            
        except Exception as e:
            await self.logger.error(f"Failed to get user sessions for {user_id}: {str(e)}")
            return []
    
    async def add_websocket_connection(self, session_id: str, connection_id: str) -> bool:
        """
        Add WebSocket connection to session.
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.add_websocket(connection_id)
            return await self.update_session(session)
            
        except Exception as e:
            await self.logger.error(f"Failed to add WebSocket to session {session_id}: {str(e)}")
            return False
    
    async def remove_websocket_connection(self, session_id: str, connection_id: str) -> bool:
        """
        Remove WebSocket connection from session.
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.remove_websocket(connection_id)
            return await self.update_session(session)
            
        except Exception as e:
            await self.logger.error(f"Failed to remove WebSocket from session {session_id}: {str(e)}")
            return False
    
    async def track_resource_usage(self, session_id: str, **usage: Any) -> bool:
        """
        Track resource usage for session.
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.update_resource_usage(**usage)
            return await self.update_session(session)
            
        except Exception as e:
            await self.logger.error(f"Failed to track resource usage for session {session_id}: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get session manager statistics.
        """
        try:
            # Get current active sessions count
            active_sessions = len([
                s for s in self._local_cache.values() 
                if s.state == SessionState.ACTIVE
            ])
            
            # Update peak if necessary
            if active_sessions > self._stats['concurrent_peak']:
                self._stats['concurrent_peak'] = active_sessions
            
            return {
                **self._stats,
                'current_active_sessions': active_sessions,
                'local_cache_size': len(self._local_cache),
                'total_locks': len(self._locks)
            }
            
        except Exception as e:
            await self.logger.error(f"Failed to get stats: {str(e)}")
            return {}
    
    async def _store_session(self, session: UserSession):
        """Store session in Redis"""
        await self.redis_cache.set(
            self._get_session_key(session.session_id),
            session.to_dict(),
            expire=int(self.session_timeout_hours * 3600)
        )
    
    async def _add_to_user_sessions(self, user_id: str, session_id: str):
        """Add session to user's session list"""
        key = self._get_user_sessions_key(user_id)
        current_sessions = await self.redis_cache.get(key) or []
        
        if session_id not in current_sessions:
            current_sessions.append(session_id)
            await self.redis_cache.set(key, current_sessions, expire=int(self.session_timeout_hours * 3600))
    
    async def _remove_from_user_sessions(self, user_id: str, session_id: str):
        """Remove session from user's session list"""
        key = self._get_user_sessions_key(user_id)
        current_sessions = await self.redis_cache.get(key) or []
        
        if session_id in current_sessions:
            current_sessions.remove(session_id)
            await self.redis_cache.set(key, current_sessions, expire=int(self.session_timeout_hours * 3600))
    
    async def _background_cleanup(self):
        """Background task for cleaning up expired sessions"""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_local_cache()
                
                # Cleanup expired sessions (implement if needed)
                # This would involve scanning Redis for expired sessions
                
                await asyncio.sleep(self.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.logger.error(f"Error in background cleanup: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def shutdown(self):
        """Shutdown the session manager"""
        try:
            self._shutdown_event.set()
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            await self.logger.info("SessionManager shutdown complete")
            
        except Exception as e:
            await self.logger.error(f"Error during SessionManager shutdown: {str(e)}")


# Singleton instance
session_manager = ConcurrentSessionManager()


async def get_session_manager() -> ConcurrentSessionManager:
    """Get the session manager instance"""
    if not hasattr(session_manager, '_initialized'):
        await session_manager.initialize()
        session_manager._initialized = True
    return session_manager


# Session-aware decorator for tracking resource usage
def track_session_usage(**usage_updates):
    """Decorator to track resource usage for session-aware operations"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Try to extract session_id from arguments
            session_id = kwargs.get('session_id')
            if not session_id and len(args) > 0:
                # Look for session_id in context or args
                context = kwargs.get('context', {})
                session_id = context.get('session_id')
            
            result = await func(*args, **kwargs)
            
            # Track resource usage if session_id is available
            if session_id:
                try:
                    manager = await get_session_manager()
                    await manager.track_resource_usage(session_id, **usage_updates)
                except Exception as e:
                    # Don't fail the operation if tracking fails
                    pass
            
            return result
        return wrapper
    return decorator
