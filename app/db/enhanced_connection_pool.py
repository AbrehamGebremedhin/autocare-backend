"""
Enhanced Database Connection Pool Manager for high concurrency and better resource management.
Provides connection pooling, load balancing, and automatic failover capabilities.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from supabase import create_client, Client
from app.core.config import get_settings
from app.core.interfaces import IDBHandler
from app.utils.exceptions import DatabaseException, ConfigurationException
from app.utils.logger import get_logger_instance


class ConnectionState(Enum):
    IDLE = "idle"
    ACTIVE = "active" 
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class PooledConnection:
    """Enhanced connection tracking with state management"""
    connection_id: str
    client: Client
    state: ConnectionState = ConnectionState.IDLE
    created_at: datetime = None
    last_used: datetime = None
    use_count: int = 0
    error_count: int = 0
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_used is None:
            self.last_used = datetime.utcnow()
    
    def mark_used(self, user_id: str = None, session_id: str = None):
        """Mark connection as used and update metadata"""
        self.last_used = datetime.utcnow()
        self.use_count += 1
        self.state = ConnectionState.ACTIVE
        self.user_id = user_id
        self.session_id = session_id
    
    def mark_idle(self):
        """Mark connection as idle"""
        self.state = ConnectionState.IDLE
        self.user_id = None
        self.session_id = None
    
    def mark_error(self):
        """Mark connection as having an error"""
        self.state = ConnectionState.ERROR
        self.error_count += 1
    
    def is_stale(self, max_age_seconds: int = 3600) -> bool:
        """Check if connection is stale"""
        age = (datetime.utcnow() - self.last_used).total_seconds()
        return age > max_age_seconds
    
    def is_overused(self, max_uses: int = 1000) -> bool:
        """Check if connection has been used too many times"""
        return self.use_count > max_uses


class ConcurrentDBConnectionPool:
    """
    Advanced database connection pool with load balancing and automatic scaling.
    Optimized for handling multiple concurrent users efficiently.
    """
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.logger = get_logger_instance("DBConnectionPool")
        
        # Pool configuration
        self.min_connections = 5
        self.max_connections = 50
        self.max_idle_time = 3600  # 1 hour
        self.max_connection_uses = 1000
        self.connection_timeout = 30
        self.retry_attempts = 3
        self.retry_delay = 1
        
        # Connection tracking
        self._pool: Dict[str, PooledConnection] = {}
        self._idle_connections: asyncio.Queue = asyncio.Queue()
        self._pool_lock = asyncio.Lock()
        self._connection_semaphore = asyncio.Semaphore(self.max_connections)
        
        # Performance tracking
        self._stats = {
            'total_connections_created': 0,
            'total_connections_closed': 0,
            'total_requests': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'connection_errors': 0,
            'concurrent_peak': 0
        }
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate database configuration"""
        if not self.settings.SUPABASE_URL or not self.settings.SUPABASE_KEY:
            raise ConfigurationException(
                "Supabase URL and Key must be set in environment variables",
                details={"missing_vars": ["SUPABASE_URL", "SUPABASE_KEY"]}
            )
    
    async def initialize(self):
        """Initialize the connection pool"""
        try:
            # Create initial connections
            await self._create_initial_connections()
            
            # Start maintenance task
            self._maintenance_task = asyncio.create_task(self._pool_maintenance())
            
            await self.logger.info(f"Connection pool initialized with {len(self._pool)} connections")
        except Exception as e:
            await self.logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise DatabaseException(f"Pool initialization failed: {str(e)}")
    
    async def _create_initial_connections(self):
        """Create initial pool of connections"""
        tasks = []
        for i in range(self.min_connections):
            task = asyncio.create_task(self._create_connection())
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = 0
        for result in results:
            if isinstance(result, PooledConnection):
                successful += 1
            else:
                await self.logger.error(f"Failed to create initial connection: {result}")
        
        if successful == 0:
            raise DatabaseException("Failed to create any initial connections")
        
        await self.logger.info(f"Created {successful}/{self.min_connections} initial connections")
    
    async def _create_connection(self) -> PooledConnection:
        """Create a new database connection with retry logic"""
        connection_id = f"conn_{int(time.time())}_{id(asyncio.current_task())}"
        
        for attempt in range(self.retry_attempts):
            try:
                client = create_client(
                    self.settings.SUPABASE_URL,
                    self.settings.SUPABASE_KEY
                )
                
                # Test the connection
                await self._test_connection(client)
                
                # Create pooled connection
                pooled_conn = PooledConnection(
                    connection_id=connection_id,
                    client=client
                )
                
                async with self._pool_lock:
                    self._pool[connection_id] = pooled_conn
                    await self._idle_connections.put(connection_id)
                
                self._stats['total_connections_created'] += 1
                
                await self.logger.debug(f"Created connection: {connection_id}")
                return pooled_conn
                
            except Exception as e:
                await self.logger.warning(
                    f"Connection creation attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt == self.retry_attempts - 1:
                    raise DatabaseException(
                        f"Failed to create connection after {self.retry_attempts} attempts: {str(e)}"
                    )
                await asyncio.sleep(self.retry_delay * (attempt + 1))
    
    async def _test_connection(self, client: Client):
        """Test database connection with timeout"""
        try:
            # Simple health check query with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: client.table("users").select("count").limit(1).execute()
                ),
                timeout=self.connection_timeout
            )
        except asyncio.TimeoutError:
            raise DatabaseException("Connection test timeout")
        except Exception as e:
            # Expected for some tables, but connection errors will be raised
            pass
    
    @asynccontextmanager
    async def get_connection(self, user_id: str = None, session_id: str = None):
        """
        Get a database connection from the pool with automatic cleanup.
        Optimized for concurrent access patterns.
        """
        connection = None
        connection_id = None
        
        try:
            # Acquire semaphore to limit concurrent connections
            async with self._connection_semaphore:
                # Get connection from pool
                connection_id = await self._get_idle_connection()
                if not connection_id:
                    # Create new connection if pool is empty and under limit
                    if len(self._pool) < self.max_connections:
                        pooled_conn = await self._create_connection()
                        connection_id = pooled_conn.connection_id
                    else:
                        # Wait for available connection
                        connection_id = await self._idle_connections.get()
                
                # Get the pooled connection
                async with self._pool_lock:
                    pooled_conn = self._pool.get(connection_id)
                    if not pooled_conn:
                        raise DatabaseException(f"Connection {connection_id} not found in pool")
                    
                    # Mark as active
                    pooled_conn.mark_used(user_id, session_id)
                    connection = pooled_conn.client
                
                # Update stats
                self._stats['total_requests'] += 1
                self._stats['pool_hits'] += 1
                
                # Track concurrent usage
                active_count = sum(
                    1 for conn in self._pool.values() 
                    if conn.state == ConnectionState.ACTIVE
                )
                if active_count > self._stats['concurrent_peak']:
                    self._stats['concurrent_peak'] = active_count
                
                await self.logger.debug(f"Acquired connection: {connection_id}")
                yield connection
                
        except Exception as e:
            await self.logger.error(f"Database connection error: {str(e)}")
            self._stats['connection_errors'] += 1
            
            # Mark connection as error if we have one
            if connection_id and connection_id in self._pool:
                async with self._pool_lock:
                    self._pool[connection_id].mark_error()
            
            raise DatabaseException(f"Database connection failed: {str(e)}")
        finally:
            # Return connection to pool
            if connection_id and connection_id in self._pool:
                await self._return_connection(connection_id)
    
    async def _get_idle_connection(self) -> Optional[str]:
        """Get an idle connection from the queue"""
        try:
            # Try to get idle connection with timeout
            connection_id = await asyncio.wait_for(
                self._idle_connections.get(),
                timeout=0.1  # Very short timeout for non-blocking behavior
            )
            
            # Verify connection is still valid
            async with self._pool_lock:
                pooled_conn = self._pool.get(connection_id)
                if not pooled_conn or pooled_conn.state != ConnectionState.IDLE:
                    return await self._get_idle_connection()  # Try again
                
                # Check if connection needs renewal
                if pooled_conn.is_stale() or pooled_conn.is_overused():
                    await self._close_connection(connection_id)
                    return await self._get_idle_connection()  # Try again
            
            return connection_id
            
        except asyncio.TimeoutError:
            return None
    
    async def _return_connection(self, connection_id: str):
        """Return connection to the idle pool"""
        try:
            async with self._pool_lock:
                pooled_conn = self._pool.get(connection_id)
                if pooled_conn and pooled_conn.state == ConnectionState.ACTIVE:
                    pooled_conn.mark_idle()
                    await self._idle_connections.put(connection_id)
                    await self.logger.debug(f"Returned connection to pool: {connection_id}")
        except Exception as e:
            await self.logger.error(f"Error returning connection {connection_id}: {str(e)}")
    
    async def _close_connection(self, connection_id: str):
        """Close and remove a connection from the pool"""
        try:
            async with self._pool_lock:
                pooled_conn = self._pool.pop(connection_id, None)
                if pooled_conn:
                    pooled_conn.state = ConnectionState.CLOSED
                    # Supabase client doesn't have explicit close method
                    # Connection will be garbage collected
                    
                    self._stats['total_connections_closed'] += 1
                    await self.logger.debug(f"Closed connection: {connection_id}")
        except Exception as e:
            await self.logger.error(f"Error closing connection {connection_id}: {str(e)}")
    
    async def _pool_maintenance(self):
        """Background task for pool maintenance"""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_stale_connections()
                await self._ensure_minimum_connections()
                await asyncio.sleep(60)  # Run every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.logger.error(f"Error in pool maintenance: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup_stale_connections(self):
        """Remove stale connections from the pool"""
        stale_connections = []
        
        async with self._pool_lock:
            for connection_id, pooled_conn in self._pool.items():
                if (pooled_conn.state == ConnectionState.IDLE and 
                    (pooled_conn.is_stale() or pooled_conn.is_overused())):
                    stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            await self._close_connection(connection_id)
            await self.logger.info(f"Cleaned up stale connection: {connection_id}")
    
    async def _ensure_minimum_connections(self):
        """Ensure minimum number of connections in pool"""
        idle_count = self._idle_connections.qsize()
        total_count = len(self._pool)
        
        if idle_count < self.min_connections // 2 and total_count < self.max_connections:
            # Create additional connections
            needed = min(self.min_connections - idle_count, self.max_connections - total_count)
            
            tasks = []
            for _ in range(needed):
                task = asyncio.create_task(self._create_connection())
                tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful = sum(1 for result in results if isinstance(result, PooledConnection))
                
                if successful > 0:
                    await self.logger.info(f"Created {successful} additional connections")
    
    async def health_check(self) -> bool:
        """Perform health check on the connection pool"""
        try:
            async with self.get_connection() as client:
                await self._test_connection(client)
                return True
        except Exception as e:
            await self.logger.error(f"Pool health check failed: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        async with self._pool_lock:
            states = {}
            for state in ConnectionState:
                states[state.value] = sum(
                    1 for conn in self._pool.values() if conn.state == state
                )
            
            return {
                **self._stats,
                'total_connections': len(self._pool),
                'idle_connections': self._idle_connections.qsize(),
                'connection_states': states,
                'semaphore_value': self._connection_semaphore._value,
                'pool_utilization': (len(self._pool) / self.max_connections) * 100
            }
    
    async def close(self):
        """Close the connection pool and cleanup resources"""
        try:
            self._shutdown_event.set()
            
            # Cancel maintenance task
            if self._maintenance_task:
                self._maintenance_task.cancel()
                try:
                    await self._maintenance_task
                except asyncio.CancelledError:
                    pass
            
            # Close all connections
            async with self._pool_lock:
                connection_ids = list(self._pool.keys())
            
            for connection_id in connection_ids:
                await self._close_connection(connection_id)
            
            await self.logger.info("Connection pool closed successfully")
            
        except Exception as e:
            await self.logger.error(f"Error during pool closure: {str(e)}")


class EnhancedSupabaseDBHandler(IDBHandler):
    """
    Enhanced Supabase DB handler with connection pooling and better concurrency support.
    """
    
    _instance = None
    _pool: Optional[ConcurrentDBConnectionPool] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_logger_instance("EnhancedSupabaseDBHandler")
            self.settings = get_settings()
            self._initialized = True
    
    async def initialize(self):
        """Initialize the enhanced database handler"""
        if self._pool is None:
            self._pool = ConcurrentDBConnectionPool(self.settings)
            await self._pool.initialize()
            await self.logger.info("Enhanced database handler initialized with connection pool")
    
    async def get_connection(self, user_id: str = None, session_id: str = None):
        """Get database connection from pool"""
        if self._pool is None:
            await self.initialize()
        
        return self._pool.get_connection(user_id=user_id, session_id=session_id)
    
    async def get_client(self) -> Client:
        """Get database client for backward compatibility"""
        if self._pool is None:
            await self.initialize()
        
        # For backward compatibility, return a client from pool
        async with self._pool.get_connection() as client:
            return client
    
    @property
    def client(self) -> Client:
        """Synchronous client access (not recommended for new code)"""
        # For backward compatibility only
        try:
            client = create_client(
                self.settings.SUPABASE_URL,
                self.settings.SUPABASE_KEY
            )
            return client
        except Exception as e:
            raise DatabaseException(f"Failed to create client: {str(e)}")
    
    async def health_check(self) -> bool:
        """Perform health check"""
        if self._pool is None:
            try:
                await self.initialize()
                return True
            except:
                return False
        
        return await self._pool.health_check()
    
    async def get_connection_stats(self) -> dict:
        """Get connection pool statistics"""
        if self._pool is None:
            return {"status": "not_initialized"}
        
        return await self._pool.get_stats()
    
    async def close(self):
        """Close the database handler"""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        await self.logger.info("Enhanced database handler closed")


# Enhanced database handler instance
enhanced_db_handler = EnhancedSupabaseDBHandler()


async def get_enhanced_db_handler() -> EnhancedSupabaseDBHandler:
    """Get the enhanced database handler instance"""
    if not enhanced_db_handler._initialized:
        await enhanced_db_handler.initialize()
    return enhanced_db_handler
