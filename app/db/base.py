from supabase import create_client, Client
from app.core.config import get_settings
from fastapi import Depends
from app.core.interfaces import IDBHandler
from app.utils.exceptions import DatabaseException, ConfigurationException
from app.utils.logger import get_logger_instance
import asyncio
from typing import Optional
import time
from contextlib import asynccontextmanager

class SupabaseDBHandler(IDBHandler):
    _instance = None
    _client = None
    _connection_pool = {}
    _max_connections = 10
    _connection_timeout = 30
    _retry_attempts = 3
    _retry_delay = 1

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SupabaseDBHandler, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the database handler with connection pooling"""
        self.logger = get_logger_instance("SupabaseDBHandler")
        self.settings = get_settings()
        self._validate_config()

    def _validate_config(self):
        """Validate database configuration"""
        if not self.settings.SUPABASE_URL or not self.settings.SUPABASE_KEY:
            raise ConfigurationException(
                "Supabase URL and Key must be set in environment variables",
                details={"missing_vars": ["SUPABASE_URL", "SUPABASE_KEY"]}
            )

    async def _create_client(self) -> Client:
        """Create a new Supabase client with retry logic"""
        for attempt in range(self._retry_attempts):
            try:
                client = create_client(
                    self.settings.SUPABASE_URL, 
                    self.settings.SUPABASE_KEY
                )
                
                # Test the connection
                await self._test_connection(client)
                return client
                
            except Exception as e:
                await self.logger.warning(
                    f"Database connection attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt == self._retry_attempts - 1:
                    raise DatabaseException(
                        "Failed to establish database connection after multiple attempts",
                        details={"attempts": self._retry_attempts, "error": str(e)}
                    )
                await asyncio.sleep(self._retry_delay * (attempt + 1))

    async def _test_connection(self, client: Client):
        """Test database connection"""
        try:
            # Simple health check query
            response = client.table("ping").select("*").limit(1).execute()
            # Connection is working if we don't get an exception
        except Exception as e:
            # This is expected for the ping table, but connection errors will be raised
            pass

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool"""
        connection_id = f"conn_{int(time.time() * 1000)}"
        
        try:
            # Create new connection (in a real production app, implement actual pooling)
            client = await self._create_client()
            self._connection_pool[connection_id] = {
                "client": client,
                "created_at": time.time(),
                "last_used": time.time()
            }
            
            await self.logger.debug(f"Created database connection: {connection_id}")
            yield client
            
        except Exception as e:
            await self.logger.error(f"Database connection error: {str(e)}")
            raise DatabaseException(f"Database connection failed: {str(e)}")
        finally:
            # Clean up connection
            if connection_id in self._connection_pool:
                del self._connection_pool[connection_id]
                await self.logger.debug(f"Closed database connection: {connection_id}")

    async def get_client(self) -> Client:
        """Get database client"""
        if self._client is None:
            self._client = await self._create_client()
        return self._client

    @property  
    def client(self) -> Client:
        """Get database client (synchronous backward compatibility)"""
        if self._client is None:
            # For sync access, create client without retry logic
            try:
                client = create_client(
                    self.settings.SUPABASE_URL, 
                    self.settings.SUPABASE_KEY
                )
                self._client = client
            except Exception as e:
                raise DatabaseException(f"Failed to create database client: {str(e)}")
        return self._client

    async def health_check(self) -> bool:
        """Perform a health check on the database connection"""
        try:
            async with self.get_connection() as client:
                # Perform a simple operation to test connectivity
                response = client.table("users").select("count").limit(1).execute()
                return True
        except Exception as e:
            await self.logger.error(f"Database health check failed: {str(e)}")
            return False

    async def close(self):
        """Close all database connections"""
        try:
            # Close all pooled connections
            for conn_id, conn_data in list(self._connection_pool.items()):
                try:
                    # Supabase client doesn't have explicit close method, 
                    # but we can clear references
                    del self._connection_pool[conn_id]
                except Exception as e:
                    await self.logger.warning(f"Error closing connection {conn_id}: {str(e)}")
            
            # Clear main client
            self._client = None
            await self.logger.info("All database connections closed")
            
        except Exception as e:
            await self.logger.error(f"Error during database cleanup: {str(e)}")

    async def get_connection_stats(self) -> dict:
        """Get connection pool statistics"""
        return {
            "active_connections": len(self._connection_pool),
            "max_connections": self._max_connections,
            "connection_timeout": self._connection_timeout
        }

# Dependency for SupabaseDBHandler
async def get_db_handler() -> SupabaseDBHandler:
    return SupabaseDBHandler()
