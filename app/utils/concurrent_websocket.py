"""
Enhanced WebSocket Manager optimized for multiple concurrent users.
Provides user-based connection management, message broadcasting, and load balancing.
"""
import asyncio
import json
import time
import uuid
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from fastapi import WebSocket
from app.core.interfaces import IWebSocketManager
from app.utils.message_formatter import MessageFormatter
from app.utils.message_types import MessageType, MessageSource
from app.utils.logger import get_logger_instance
from app.utils.session_manager import get_session_manager, ConcurrentSessionManager


@dataclass
class WebSocketConnection:
    """Enhanced WebSocket connection tracking"""
    connection_id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    connected_at: datetime = None
    last_activity: datetime = None
    message_count: int = 0
    connection_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.connected_at is None:
            self.connected_at = datetime.utcnow()
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()
        if self.connection_metadata is None:
            self.connection_metadata = {}
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
        self.message_count += 1


class ConcurrentWebSocketManager(IWebSocketManager):
    """
    Enhanced WebSocket manager optimized for high concurrency and multiple users.
    Provides proper connection isolation, user-based broadcasting, and resource tracking.
    """
    
    def __init__(self, session_manager: Optional[ConcurrentSessionManager] = None):
        self.logger = get_logger_instance("WebSocketManager")
        self.session_manager = session_manager
        
        # Connection tracking
        self._connections: Dict[str, WebSocketConnection] = {}
        self._user_connections: Dict[str, Set[str]] = defaultdict(set)
        self._session_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Performance tracking
        self._stats = {
            'total_connections': 0,
            'total_disconnections': 0,
            'total_messages_sent': 0,
            'total_broadcasts': 0,
            'concurrent_peak': 0,
            'connection_errors': 0
        }
        
        # Configuration
        self.max_connections_per_user = 5
        self.connection_timeout = 3600  # 1 hour
        self.heartbeat_interval = 30  # 30 seconds
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize the WebSocket manager"""
        try:
            if self.session_manager is None:
                self.session_manager = await get_session_manager()
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
            await self.logger.info("ConcurrentWebSocketManager initialized")
        except Exception as e:
            await self.logger.error(f"Failed to initialize WebSocketManager: {str(e)}")
            raise
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """
        Enhanced connect method with user and session tracking.
        """
        try:
            await websocket.accept()
            
            # Generate unique connection ID
            connection_id = f"ws_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            async with self._lock:
                # Check connection limits for user
                if user_id and len(self._user_connections[user_id]) >= self.max_connections_per_user:
                    # Disconnect oldest connection for this user
                    oldest_conn_id = min(
                        self._user_connections[user_id],
                        key=lambda cid: self._connections[cid].connected_at
                    )
                    await self._force_disconnect(oldest_conn_id, "Connection limit exceeded")
                
                # Create connection object
                connection = WebSocketConnection(
                    connection_id=connection_id,
                    websocket=websocket,
                    user_id=user_id,
                    session_id=session_id
                )
                
                # Store connection
                self._connections[connection_id] = connection
                
                # Update user and session mappings
                if user_id:
                    self._user_connections[user_id].add(connection_id)
                if session_id:
                    self._session_connections[session_id].add(connection_id)
                
                # Update stats
                self._stats['total_connections'] += 1
                current_count = len(self._connections)
                if current_count > self._stats['concurrent_peak']:
                    self._stats['concurrent_peak'] = current_count
                
                # Update session manager
                if session_id and self.session_manager:
                    await self.session_manager.add_websocket_connection(session_id, connection_id)
            
            await self.logger.info(f"WebSocket connected: {connection_id} (user: {user_id}, session: {session_id})")
            
            # Store connection ID in websocket state for cleanup
            websocket.state.connection_id = connection_id
            
            # Return connection information
            return {
                "connection_id": connection_id,
                "user_id": user_id,
                "session_id": session_id,
                "connected_at": connection.connected_at
            }
            
        except Exception as e:
            await self.logger.error(f"Failed to establish WebSocket connection: {str(e)}")
            self._stats['connection_errors'] += 1
            raise
    
    async def disconnect(self, websocket: WebSocket):
        """
        Enhanced disconnect method with proper cleanup.
        """
        try:
            connection_id = getattr(websocket.state, 'connection_id', None)
            if not connection_id:
                # Try to find connection by websocket object
                connection_id = next(
                    (cid for cid, conn in self._connections.items() if conn.websocket == websocket),
                    None
                )
            
            if connection_id:
                await self._remove_connection(connection_id)
            
        except Exception as e:
            await self.logger.error(f"Error during WebSocket disconnect: {str(e)}")
    
    async def _remove_connection(self, connection_id: str):
        """Remove connection and cleanup associated data"""
        async with self._lock:
            connection = self._connections.get(connection_id)
            if not connection:
                return
            
            # Remove from mappings
            if connection.user_id:
                self._user_connections[connection.user_id].discard(connection_id)
                if not self._user_connections[connection.user_id]:
                    del self._user_connections[connection.user_id]
            
            if connection.session_id:
                self._session_connections[connection.session_id].discard(connection_id)
                if not self._session_connections[connection.session_id]:
                    del self._session_connections[connection.session_id]
                
                # Update session manager
                if self.session_manager:
                    await self.session_manager.remove_websocket_connection(
                        connection.session_id, connection_id
                    )
            
            # Remove from connections
            del self._connections[connection_id]
            
            # Update stats
            self._stats['total_disconnections'] += 1
            
            await self.logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def _force_disconnect(self, connection_id: str, reason: str = "Forced disconnect"):
        """Force disconnect a specific connection"""
        try:
            connection = self._connections.get(connection_id)
            if connection:
                # Send disconnect message
                try:
                    await connection.websocket.send_text(json.dumps({
                        "type": "disconnect",
                        "reason": reason,
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                except:
                    pass  # Connection might already be closed
                
                # Close connection
                try:
                    await connection.websocket.close(code=1000, reason=reason)
                except:
                    pass
                
                # Remove from tracking
                await self._remove_connection(connection_id)
                
        except Exception as e:
            await self.logger.error(f"Error force disconnecting {connection_id}: {str(e)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(message)
            self._stats['total_messages_sent'] += 1
            
            # Update connection activity
            connection_id = getattr(websocket.state, 'connection_id', None)
            if connection_id and connection_id in self._connections:
                self._connections[connection_id].update_activity()
                
        except Exception as e:
            await self.logger.error(f"Failed to send personal message: {str(e)}")
            # Remove dead connection
            await self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all active connections"""
        try:
            async with self._lock:
                connections = list(self._connections.values())
            
            # Send to all connections concurrently
            tasks = []
            for connection in connections:
                task = asyncio.create_task(self._safe_send(connection, message))
                tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count successful sends
                successful = sum(1 for result in results if result is True)
                failed = len(results) - successful
                
                if failed > 0:
                    await self.logger.warning(f"Broadcast partially failed: {failed}/{len(results)} failed")
                
                self._stats['total_broadcasts'] += 1
                self._stats['total_messages_sent'] += successful
            
        except Exception as e:
            await self.logger.error(f"Broadcast failed: {str(e)}")
    
    async def broadcast_to_user(self, user_id: str, message: str):
        """Broadcast message to all connections of a specific user"""
        try:
            connection_ids = self._user_connections.get(user_id, set()).copy()
            
            if not connection_ids:
                return 0  # No connections for this user
            
            # Send to all user connections concurrently
            tasks = []
            for connection_id in connection_ids:
                connection = self._connections.get(connection_id)
                if connection:
                    task = asyncio.create_task(self._safe_send(connection, message))
                    tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful = sum(1 for result in results if result is True)
                self._stats['total_messages_sent'] += successful
                return successful
            
            return 0
            
        except Exception as e:
            await self.logger.error(f"Failed to broadcast to user {user_id}: {str(e)}")
            return 0
    
    async def broadcast_to_session(self, session_id: str, message: str):
        """Broadcast message to all connections of a specific session"""
        try:
            connection_ids = self._session_connections.get(session_id, set()).copy()
            
            if not connection_ids:
                return 0  # No connections for this session
            
            # Send to all session connections concurrently
            tasks = []
            for connection_id in connection_ids:
                connection = self._connections.get(connection_id)
                if connection:
                    task = asyncio.create_task(self._safe_send(connection, message))
                    tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful = sum(1 for result in results if result is True)
                self._stats['total_messages_sent'] += successful
                return successful
            
            return 0
            
        except Exception as e:
            await self.logger.error(f"Failed to broadcast to session {session_id}: {str(e)}")
            return 0
    
    async def _safe_send(self, connection: WebSocketConnection, message: str) -> bool:
        """Safely send message to connection with error handling"""
        try:
            await connection.websocket.send_text(message)
            connection.update_activity()
            return True
        except Exception as e:
            # Connection is dead, remove it
            await self._remove_connection(connection.connection_id)
            return False
    
    # IWebSocketManager interface implementations
    async def send_info(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.INFO, source=source, content=content, session_id=session_id, details=details)
        await self.send_personal_message(json.dumps(msg), websocket)

    async def send_warning(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.WARNING, source=source, content=content, session_id=session_id, details=details)
        await self.send_personal_message(json.dumps(msg), websocket)

    async def send_error(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.ERROR, source=source, content=content, session_id=session_id, details=details)
        await self.send_personal_message(json.dumps(msg), websocket)

    async def send_progress(self, websocket: WebSocket, content: str, source: MessageSource, progress: float, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.PROGRESS, source=source, content=content, progress=progress, session_id=session_id, details=details)
        await self.send_personal_message(json.dumps(msg), websocket)

    async def send_stage(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.STAGE, source=source, content=content, session_id=session_id, details=details)
        await self.send_personal_message(json.dumps(msg), websocket)

    async def send_result(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.RESULT, source=source, content=content, session_id=session_id, details=details)
        await self.send_personal_message(json.dumps(msg), websocket)

    async def send_debug(self, websocket: WebSocket, content: str, source: MessageSource, session_id: str = None, details: dict = None):
        msg = MessageFormatter.format(type=MessageType.DEBUG, source=source, content=content, session_id=session_id, details=details)
        await self.send_personal_message(json.dumps(msg), websocket)

    async def broadcast_json(self, msg: dict):
        await self.broadcast(json.dumps(msg))

    async def broadcast_standard(self, *, type: MessageType, source: MessageSource, content: str, session_id: str = None, progress: float = None, details: dict = None):
        msg = MessageFormatter.format(type=type, source=source, content=content, session_id=session_id, progress=progress, details=details)
        await self.broadcast(json.dumps(msg))
    
    # Enhanced methods for user/session-specific broadcasting
    async def broadcast_to_user_standard(self, user_id: str, *, type: MessageType, source: MessageSource, content: str, session_id: str = None, progress: float = None, details: dict = None):
        """Broadcast structured message to all user connections"""
        msg = MessageFormatter.format(type=type, source=source, content=content, session_id=session_id, progress=progress, details=details)
        return await self.broadcast_to_user(user_id, json.dumps(msg))
    
    async def broadcast_to_session_standard(self, session_id: str, *, type: MessageType, source: MessageSource, content: str, progress: float = None, details: dict = None):
        """Broadcast structured message to all session connections"""
        msg = MessageFormatter.format(type=type, source=source, content=content, session_id=session_id, progress=progress, details=details)
        return await self.broadcast_to_session(session_id, json.dumps(msg))
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics"""
        async with self._lock:
            user_counts = {user_id: len(connections) for user_id, connections in self._user_connections.items()}
            session_counts = {session_id: len(connections) for session_id, connections in self._session_connections.items()}
            
            # Calculate activity metrics
            now = datetime.utcnow()
            active_connections = 0
            idle_connections = 0
            
            for connection in self._connections.values():
                time_since_activity = (now - connection.last_activity).total_seconds()
                if time_since_activity < 300:  # 5 minutes
                    active_connections += 1
                else:
                    idle_connections += 1
            
            return {
                **self._stats,
                'current_connections': len(self._connections),
                'active_connections': active_connections,
                'idle_connections': idle_connections,
                'unique_users': len(self._user_connections),
                'unique_sessions': len(self._session_connections),
                'avg_connections_per_user': sum(user_counts.values()) / len(user_counts) if user_counts else 0,
                'max_connections_per_user': max(user_counts.values()) if user_counts else 0,
                'user_distribution': user_counts,
                'session_distribution': session_counts
            }
    
    async def _background_cleanup(self):
        """Background task for cleaning up stale connections"""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_stale_connections()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.logger.error(f"Error in background cleanup: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup_stale_connections(self):
        """Remove connections that haven't been active for too long"""
        now = datetime.utcnow()
        stale_threshold = self.connection_timeout
        stale_connections = []
        
        async with self._lock:
            for connection_id, connection in self._connections.items():
                time_since_activity = (now - connection.last_activity).total_seconds()
                if time_since_activity > stale_threshold:
                    stale_connections.append(connection_id)
        
        # Remove stale connections
        for connection_id in stale_connections:
            await self._force_disconnect(connection_id, "Connection timeout")
            await self.logger.info(f"Cleaned up stale connection: {connection_id}")
    
    async def _heartbeat_monitor(self):
        """Send periodic heartbeat to all connections"""
        while not self._shutdown_event.is_set():
            try:
                heartbeat_msg = json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "server_time": int(time.time())
                })
                
                await self.broadcast(heartbeat_msg)
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.logger.error(f"Error in heartbeat monitor: {str(e)}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def close(self):
        """Shutdown the WebSocket manager"""
        try:
            self._shutdown_event.set()
            
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            
            # Close all connections
            async with self._lock:
                connection_ids = list(self._connections.keys())
            
            for connection_id in connection_ids:
                await self._force_disconnect(connection_id, "Server shutdown")
            
            await self.logger.info("ConcurrentWebSocketManager shutdown complete")
            
        except Exception as e:
            await self.logger.error(f"Error during WebSocketManager shutdown: {str(e)}")


# Enhanced manager instance
concurrent_websocket_manager = ConcurrentWebSocketManager()


async def get_websocket_manager() -> ConcurrentWebSocketManager:
    """Get the enhanced WebSocket manager instance"""
    if not hasattr(concurrent_websocket_manager, '_initialized'):
        await concurrent_websocket_manager.initialize()
        concurrent_websocket_manager._initialized = True
    return concurrent_websocket_manager
