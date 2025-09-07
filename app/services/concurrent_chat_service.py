"""
Enhanced Chat Service optimized for multiple concurrent users.
Provides proper session isolation, caching, and horizontal scaling support.
"""
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from uuid import uuid4
import logging

from app.services.base_service import BaseService
from app.agents.orchestrator_agent import OrchestratorAgent
from app.utils.diagnosis_tree import DiagnosisTreeNode
from app.core.interfaces import IWebSocketManager
from app.utils.message_types import MessageSource
from app.utils.redis_cache import RedisCache, get_redis_cache
from app.utils.session_manager import get_session_manager, ConcurrentSessionManager, track_session_usage
from app.utils.logger import get_logger_instance
from app.utils.exceptions import ServiceException, ValidationException


class ConcurrentChatService(BaseService):
    """
    Enhanced ChatService optimized for multiple concurrent users.
    Features:
    - Redis-based session storage for horizontal scaling
    - Session isolation and proper resource management
    - Enhanced caching and performance optimization
    - Real-time WebSocket support with user/session targeting
    """
    
    def __init__(
        self,
        websocket_manager: IWebSocketManager = None,
        orchestrator: Optional[OrchestratorAgent] = None,
        logger: Optional[logging.Logger] = None,
        redis_cache: Optional[RedisCache] = None,
        session_manager: Optional[ConcurrentSessionManager] = None,
        max_conversation_length: int = 20,
        conversation_ttl: int = 3600,
    ):
        super().__init__(websocket_manager=websocket_manager)
        self.orchestrator = orchestrator or OrchestratorAgent()
        self.logger = logger or get_logger_instance("ConcurrentChatService")
        self.redis_cache = redis_cache
        self.session_manager = session_manager
        
        # Configuration
        self._max_conversation_length = max_conversation_length
        self._conversation_ttl = conversation_ttl
        
        # Local cache for frequently accessed conversations (with TTL)
        self._local_cache: Dict[str, Dict] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._local_cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self._response_times = deque(maxlen=1000)  # Track last 1000 responses
        self._stats = {
            'total_messages': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0.0,
            'concurrent_conversations': 0,
            'sessions_created': 0,
            'errors': 0
        }
        
        # Rate limiting per user
        self._user_rate_limits: Dict[str, deque] = {}
        self._rate_limit_window = 60  # 1 minute
        self._max_requests_per_minute = 30
        
        # Locks for thread safety
        self._conversation_locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the chat service"""
        try:
            if self.redis_cache is None:
                self.redis_cache = await get_redis_cache()
            
            if self.session_manager is None:
                self.session_manager = await get_session_manager()
            
            await self.logger.info("ConcurrentChatService initialized successfully")
        except Exception as e:
            await self.logger.error(f"Failed to initialize ConcurrentChatService: {str(e)}")
            raise ServiceException(f"Initialization failed: {str(e)}")
    
    def _get_conversation_key(self, user_id: str, session_id: str = None) -> str:
        """Get Redis key for conversation storage"""
        if session_id:
            return f"conversation:session:{session_id}"
        return f"conversation:user:{user_id}"
    
    async def _get_conversation_lock(self, conversation_key: str) -> asyncio.Lock:
        """Get or create lock for conversation-specific operations"""
        async with self._global_lock:
            if conversation_key not in self._conversation_locks:
                self._conversation_locks[conversation_key] = asyncio.Lock()
            return self._conversation_locks[conversation_key]
    
    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits"""
        now = time.time()
        
        if user_id not in self._user_rate_limits:
            self._user_rate_limits[user_id] = deque()
        
        requests = self._user_rate_limits[user_id]
        
        # Remove old requests outside the window
        while requests and now - requests[0] > self._rate_limit_window:
            requests.popleft()
        
        # Check if under limit
        if len(requests) >= self._max_requests_per_minute:
            await self.logger.warning(f"Rate limit exceeded for user {user_id}")
            return False
        
        # Add current request
        requests.append(now)
        return True
    
    async def _get_conversation_from_cache(self, conversation_key: str) -> Optional[Dict]:
        """Get conversation from local cache"""
        if conversation_key in self._local_cache:
            cached_time = self._cache_timestamps.get(conversation_key, 0)
            if time.time() - cached_time < self._local_cache_ttl:
                self._stats['cache_hits'] += 1
                return self._local_cache[conversation_key]
            else:
                # Remove expired cache entry
                self._local_cache.pop(conversation_key, None)
                self._cache_timestamps.pop(conversation_key, None)
        
        return None
    
    async def _cache_conversation(self, conversation_key: str, conversation: Dict):
        """Cache conversation locally"""
        self._local_cache[conversation_key] = conversation
        self._cache_timestamps[conversation_key] = time.time()
    
    async def _get_conversation(self, user_id: str, session: Optional[Dict] = None) -> Dict:
        """
        Get conversation with enhanced caching and session management.
        Optimized for concurrent access.
        """
        session_id = session.get('id') if session else None
        conversation_key = self._get_conversation_key(user_id, session_id)
        
        # Try local cache first
        conversation = await self._get_conversation_from_cache(conversation_key)
        if conversation:
            return conversation
        
        # Cache miss - get lock for this conversation
        lock = await self._get_conversation_lock(conversation_key)
        
        async with lock:
            # Double-check cache after acquiring lock
            conversation = await self._get_conversation_from_cache(conversation_key)
            if conversation:
                return conversation
            
            self._stats['cache_misses'] += 1
            
            # Try Redis cache
            try:
                conversation_data = await self.redis_cache.get(conversation_key)
                if conversation_data:
                    conversation = conversation_data
                    # Validate conversation structure
                    if self._validate_conversation(conversation):
                        await self._cache_conversation(conversation_key, conversation)
                        return conversation
            except Exception as e:
                await self.logger.warning(f"Failed to get conversation from Redis: {str(e)}")
            
            # Use provided session data
            if session:
                conversation = await self._process_session_data(session)
                # Cache the processed conversation
                await self._store_conversation(conversation_key, conversation)
                await self._cache_conversation(conversation_key, conversation)
                return conversation
            
            # Create new conversation
            conversation = await self._create_new_conversation(user_id, session_id)
            await self._store_conversation(conversation_key, conversation)
            await self._cache_conversation(conversation_key, conversation)
            
            self._stats['sessions_created'] += 1
            return conversation
    
    def _validate_conversation(self, conversation: Dict) -> bool:
        """Validate conversation structure"""
        required_fields = ['id', 'user_id', 'messages', 'created_at', 'context']
        return all(field in conversation for field in required_fields)
    
    async def _process_session_data(self, session: Dict) -> Dict:
        """Process session data to ensure proper format"""
        now = datetime.now()
        
        # Ensure all required fields exist
        if 'last_updated' not in session:
            session['last_updated'] = now.isoformat()
        
        # Ensure messages field exists
        if 'messages' not in session:
            session['messages'] = []
            
        # Ensure basic fields exist for minimal sessions
        if 'user_id' not in session:
            session['user_id'] = 'anonymous'
        if 'created_at' not in session:
            session['created_at'] = now.isoformat()
        if 'updated_at' not in session:
            session['updated_at'] = now.isoformat()
        
        # Limit message history
        if len(session.get('messages', [])) > self._max_conversation_length:
            session['messages'] = session['messages'][-self._max_conversation_length:]
        
        # Process diagnosis tree
        tree = session.get('diagnosis_tree')
        ctx = session.get('context', {})
        if not isinstance(ctx, dict):
            ctx = {}
        
        if tree and isinstance(tree, (dict, list)):
            try:
                tree_data = tree[0] if isinstance(tree, list) and tree else tree
                from app.schemas.Chat_Session import ChatSession
                ctx['diagnosis_tree'] = ChatSession.deserialize_diagnosis_tree(tree_data)
            except Exception as e:
                await self.logger.warning(f"Failed to deserialize diagnosis tree: {str(e)}")
                from app.utils.diagnosis_tree import DiagnosisTreeNode
                ctx['diagnosis_tree'] = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
        elif not ctx.get('diagnosis_tree'):
            from app.utils.diagnosis_tree import DiagnosisTreeNode
            ctx['diagnosis_tree'] = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
        
        session['context'] = ctx
        
        # Ensure title exists
        if 'title' not in session or not session['title']:
            session['title'] = 'New Chat Session'
        
        return session
    
    async def _create_new_conversation(self, user_id: str, session_id: str = None) -> Dict:
        """Create a new conversation"""
        if not session_id:
            session_id = str(uuid4())
        
        now = datetime.now()
        from app.utils.diagnosis_tree import DiagnosisTreeNode
        
        diagnosis_tree = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
        
        conversation = {
            'id': session_id,
            'user_id': user_id,
            'title': 'New Chat Session',
            'messages': [],
            'created_at': now.isoformat(),
            'updated_at': now.isoformat(),
            'context': {'diagnosis_tree': diagnosis_tree},
        }
        
        return conversation
    
    async def _store_conversation(self, conversation_key: str, conversation: Dict):
        """Store conversation in Redis"""
        try:
            # Prepare conversation for storage (exclude non-serializable objects)
            storage_conversation = self._prepare_for_storage(conversation)
            
            await self.redis_cache.set(
                conversation_key,
                storage_conversation,
                expire=self._conversation_ttl
            )
        except Exception as e:
            await self.logger.error(f"Failed to store conversation: {str(e)}")
    
    def _prepare_for_storage(self, conversation: Dict) -> Dict:
        """Prepare conversation for Redis storage"""
        storage_conv = conversation.copy()
        
        # Handle diagnosis tree serialization
        context = storage_conv.get('context', {})
        if 'diagnosis_tree' in context:
            tree = context['diagnosis_tree']
            if hasattr(tree, 'to_dict'):
                context['diagnosis_tree'] = tree.to_dict()
            elif hasattr(tree, '__dict__'):
                # Fallback serialization
                context['diagnosis_tree'] = {
                    'issue_name': getattr(tree, 'issue_name', 'root'),
                    'likelyhood': getattr(tree, 'likelyhood', 1.0),
                    'children': []
                }
        
        storage_conv['context'] = context
        return storage_conv
    
    @track_session_usage(llm_tokens=100, db_queries=1, websocket_messages=1)
    async def send_message(
        self, 
        user_id: str, 
        message: str, 
        context: Optional[Dict] = None, 
        session: Optional[Dict] = None, 
        websocket=None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Enhanced send_message with better concurrency and session management.
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if not user_id or len(user_id) > 100:
                raise ValidationException("Invalid user_id")
            
            if not message or len(message) > 10000:
                raise ValidationException("Invalid message length")
            
            # Check rate limits
            if not await self._check_rate_limit(user_id):
                return {
                    'response': "Rate limit exceeded. Please wait before sending another message.",
                    'confidence': 0.0,
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'rate_limited': True
                }
            
            # Get or create session in session manager
            if session_id and self.session_manager:
                user_session = await self.session_manager.get_session(session_id)
                if not user_session:
                    # Create new session
                    user_session = await self.session_manager.create_session(
                        user_id=user_id,
                        chat_session_id=session_id,
                        metadata={'chat_active': True}
                    )
            
            # Get conversation with enhanced caching
            conversation = await self._get_conversation(user_id, session=session)
            is_initial = len(conversation['messages']) == 0
            
            # Handle initial message context
            if is_initial:
                await self._handle_initial_message_context(conversation, message, context)
            else:
                await self._handle_follow_up_message_context(conversation, message, context)
            
            # Add user message to conversation
            message_context = {k: v for k, v in (context or {}).items() if k != 'diagnosis_tree'}
            conversation['messages'].append({
                'role': 'user',
                'content': message,
                'timestamp': start_time.isoformat(),
                'context': message_context,
                'is_initial': is_initial
            })
            
            # Prepare context for orchestrator
            orchestrator_context = dict(context or {})
            orchestrator_context.update({
                'diagnosis_tree': conversation['context'].get('diagnosis_tree'),
                'conversation_history': conversation['messages'],
                'is_initial_message': is_initial
            })
            
            # Send progress notification
            if websocket:
                try:
                    await self.send_ws_progress(
                        websocket, 
                        "Processing message", 
                        MessageSource.CHAT_SERVICE, 
                        0.1, 
                        session_id=conversation['id']
                    )
                except Exception as ws_error:
                    await self.logger.warning(f"WebSocket notification failed: {ws_error}")
            
            # Send to orchestrator
            response_data = await self.orchestrator.route_request(
                message,
                user_id,
                orchestrator_context,
                websocket=websocket,
                session_id=conversation['id']
            )
            
            # Handle orchestrator response
            if response_data is None:
                await self.logger.warning("Orchestrator returned None response")
                response_data = {
                    'response': "I apologize, but I couldn't process your request at this time. Please try again.",
                    'confidence': 0.0,
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Update conversation with response
            await self._update_conversation_with_response(conversation, response_data)
            
            # Store updated conversation
            conversation_key = self._get_conversation_key(user_id, conversation['id'])
            await self._store_conversation(conversation_key, conversation)
            await self._cache_conversation(conversation_key, conversation)
            
            # Track performance
            response_time = (datetime.now() - start_time).total_seconds()
            self._response_times.append(response_time)
            self._stats['total_messages'] += 1
            
            # Update response data
            response_data.update({
                'response_time': response_time,
                'conversation_length': len(conversation['messages']),
                'session_id': conversation['id']
            })
            
            # Send final notification
            if websocket:
                try:
                    await self.send_ws_result(
                        websocket,
                        "Message processed successfully",
                        MessageSource.CHAT_SERVICE,
                        session_id=conversation['id'],
                        details=response_data
                    )
                except Exception as ws_error:
                    await self.logger.warning(f"WebSocket final notification failed: {ws_error}")
            
            return response_data
            
        except ValidationException as ve:
            await self.logger.warning(f"Validation error for user {user_id}: {str(ve)}")
            return {
                'response': str(ve),
                'confidence': 0.0,
                'error': 'validation_error',
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            await self.logger.error(f"Error in send_message for user {user_id}: {str(e)}")
            self._stats['errors'] += 1
            
            if websocket:
                try:
                    await self.send_ws_error(
                        websocket,
                        "Error processing message",
                        MessageSource.CHAT_SERVICE,
                        session_id=session_id,
                        details={"error": str(e)}
                    )
                except Exception as ws_error:
                    await self.logger.warning(f"WebSocket error notification failed: {ws_error}")
            
            return {
                'response': "I'm experiencing some technical difficulties. Please try again in a moment.",
                'confidence': 0.0,
                'error': str(e),
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _handle_initial_message_context(self, conversation: Dict, message: str, context: Optional[Dict]):
        """Handle context for initial messages"""
        try:
            from app.agents.session_context_agent import SessionContextAgent, OriginalIssueContext
            
            context_agent = SessionContextAgent(
                car_id=self._get_car_id_from_context(context),
                car_make=self._get_car_make_from_context(context),
                car_model=self._get_car_model_from_context(context),
                car_year=self._get_car_year_from_context(context)
            )
            
            original_context = await context_agent.extract_original_issue(
                message,
                conversation['id'],
                self._get_car_make_from_context(context) or "",
                self._get_car_model_from_context(context) or "",
                self._get_car_year_from_context(context) or ""
            )
            
            conversation['context']['original_issue'] = original_context.to_dict()
            
        except Exception as e:
            await self.logger.warning(f"Failed to extract initial context: {str(e)}")
    
    async def _handle_follow_up_message_context(self, conversation: Dict, message: str, context: Optional[Dict]):
        """Handle context for follow-up messages"""
        try:
            if 'original_issue' in conversation['context']:
                from app.agents.session_context_agent import SessionContextAgent, OriginalIssueContext
                
                context_agent = SessionContextAgent(
                    car_id=self._get_car_id_from_context(context),
                    car_make=self._get_car_make_from_context(context),
                    car_model=self._get_car_model_from_context(context),
                    car_year=self._get_car_year_from_context(context)
                )
                
                original_issue_data = conversation['context']['original_issue']
                if original_issue_data:
                    original_context = OriginalIssueContext.from_dict(original_issue_data)
                    if original_context:
                        context_agent.original_issue_contexts[conversation['id']] = original_context
                        
                        is_relevant = await context_agent.is_message_relevant(message, conversation['id'])
                        
                        if not is_relevant:
                            await self.logger.info(f"Off-topic message detected for session {conversation['id']}")
                            # Could implement topic redirection here
        except Exception as e:
            await self.logger.warning(f"Failed to handle follow-up context: {str(e)}")
    
    async def _update_conversation_with_response(self, conversation: Dict, response_data: Dict):
        """Update conversation with orchestrator response"""
        # Update diagnosis tree if returned
        if 'diagnosis_tree' in response_data:
            conversation['context']['diagnosis_tree'] = response_data['diagnosis_tree']
            
            # Update session title if it's still default
            current_title = conversation.get('title', 'New Chat Session')
            if current_title == 'New Chat Session':
                try:
                    from app.schemas.Chat_Session import ChatSession
                    new_title = ChatSession.generate_session_title(
                        conversation['context']['diagnosis_tree'],
                        conversation['messages']
                    )
                    conversation['title'] = new_title
                except Exception as e:
                    await self.logger.warning(f"Failed to generate session title: {str(e)}")
        
        # Add assistant response
        conversation['messages'].append({
            'role': 'assistant',
            'content': response_data.get('response', ''),
            'timestamp': datetime.now().isoformat(),
            'confidence': response_data.get('confidence', 0.0),
            'sources': response_data.get('sources', [])
        })
        
        # Update timestamps
        conversation['updated_at'] = datetime.now().isoformat()
        if isinstance(conversation.get('created_at'), datetime):
            conversation['created_at'] = conversation['created_at'].isoformat()
    
    async def get_user_conversations(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversations for a user"""
        try:
            # This would typically query a database for user's conversations
            # For now, we'll return empty list as this requires database integration
            return []
        except Exception as e:
            await self.logger.error(f"Failed to get user conversations: {str(e)}")
            return []
    
    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation"""
        try:
            conversation_key = self._get_conversation_key(user_id, conversation_id)
            
            # Remove from Redis
            await self.redis_cache.delete(conversation_key)
            
            # Remove from local cache
            self._local_cache.pop(conversation_key, None)
            self._cache_timestamps.pop(conversation_key, None)
            
            # Remove conversation lock
            self._conversation_locks.pop(conversation_key, None)
            
            return True
        except Exception as e:
            await self.logger.error(f"Failed to delete conversation {conversation_id}: {str(e)}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if self._response_times:
            avg_response_time = sum(self._response_times) / len(self._response_times)
            min_response_time = min(self._response_times)
            max_response_time = max(self._response_times)
        else:
            avg_response_time = min_response_time = max_response_time = 0.0
        
        return {
            **self._stats,
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'local_cache_size': len(self._local_cache),
            'active_conversations': len(self._conversation_locks),
            'response_count': len(self._response_times)
        }
    
    async def perform_action(self, *args, **kwargs):
        """BaseService compatibility method"""
        return await self.run_with_notification(self._perform_action_impl, *args, **kwargs)
    
    async def _perform_action_impl(self, *args, **kwargs):
        """Implementation for BaseService compatibility"""
        pass
    
    # Helper methods for context extraction (same as original)
    def _get_car_id_from_context(self, context: Optional[Dict]) -> Optional[str]:
        if not context:
            return None
        if 'car' in context:
            return context['car'].get('id')
        if 'original_issue' in context:
            return context['original_issue'].get('car_id')
        return None
    
    def _get_car_make_from_context(self, context: Optional[Dict]) -> Optional[str]:
        if not context:
            return None
        if 'car' in context:
            return context['car'].get('make')
        if 'original_issue' in context:
            return context['original_issue'].get('car_make')
        return None
    
    def _get_car_model_from_context(self, context: Optional[Dict]) -> Optional[str]:
        if not context:
            return None
        if 'car' in context:
            return context['car'].get('model')
        if 'original_issue' in context:
            return context['original_issue'].get('car_model')
        return None
    
    def _get_car_year_from_context(self, context: Optional[Dict]) -> Optional[int]:
        if not context:
            return None
        if 'car' in context:
            year = context['car'].get('year')
            return int(year) if year else None
        if 'original_issue' in context:
            year = context['original_issue'].get('car_year')
            return int(year) if year else None
        return None


# Enhanced chat service instance
concurrent_chat_service = ConcurrentChatService()


async def get_concurrent_chat_service() -> ConcurrentChatService:
    """Get the enhanced chat service instance"""
    if not hasattr(concurrent_chat_service, '_initialized'):
        await concurrent_chat_service.initialize()
        concurrent_chat_service._initialized = True
    return concurrent_chat_service
