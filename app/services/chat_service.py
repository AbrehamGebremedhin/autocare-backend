from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from app.services.base_service import BaseService
from app.agents.orchestrator_agent import OrchestratorAgent
from collections import deque
from uuid import uuid4

class ChatService(BaseService):
    """
    ChatService with session management and direct message passing to OrchestratorAgent.
    """
    def __init__(self, websocket_manager=None):
        super().__init__(websocket_manager=websocket_manager)
        self.orchestrator = OrchestratorAgent()
        self._conversation_cache = {}  # Cache for active conversations
        self._max_conversation_length = 20  # Maximum messages to keep in memory
        self._conversation_ttl = 3600  # 1 hour TTL for conversations
        self._response_times = deque(maxlen=100)  # Track last 100 response times

    def _get_conversation(self, user_id: str) -> Dict:
        now = datetime.now()
        if user_id in self._conversation_cache:
            conversation = self._conversation_cache[user_id]
            last_updated = conversation.get('last_updated', now)
            if (now - last_updated).total_seconds() < self._conversation_ttl:
                if len(conversation['messages']) > self._max_conversation_length:
                    conversation['messages'] = conversation['messages'][-self._max_conversation_length:]
                return conversation
        # Create new conversation
        session_id = str(uuid4())
        conversation = {
            'id': session_id,
            'user_id': user_id,
            'messages': [],
            'created_at': now,
            'updated_at': now,
            'context': {},
        }
        self._conversation_cache[user_id] = conversation
        return conversation

    @BaseService.cache_result(ttl_seconds=300)
    async def send_message(self, user_id: str, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Send a message from a user and return the response from the OrchestratorAgent.
        Args:
            user_id: Unique identifier for the user
            message: User's message
            context: Optional context information (car details, symptoms, etc.)
        Returns:
            Dict containing response, confidence, sources, and metadata
        """
        start_time = datetime.now()
        
        try:
            await self._rate_limit()  # Apply rate limiting
            
            conversation = self._get_conversation(user_id)
            is_initial = len(conversation['messages']) == 0
            # Add user message to conversation
            conversation['messages'].append({
                'role': 'user',
                'content': message,
                'timestamp': start_time.isoformat(),
                'context': context,
                'is_initial': is_initial
            })
            # Mark initial message in context for orchestrator
            if context is None:
                context = {}
            if is_initial:
                context = dict(context)
                context['is_initial_message'] = True
            response_data = await self.orchestrator.route_request(message, user_id, context)
            # Add assistant response to conversation
            conversation['messages'].append({
                'role': 'assistant',
                'content': response_data.get('response', ''),
                'timestamp': datetime.now().isoformat(),
                'confidence': response_data.get('confidence', 0.0),
                'sources': response_data.get('sources', [])
            })
            conversation['last_updated'] = datetime.now()
            self._conversation_cache[user_id] = conversation
            response_time = (datetime.now() - start_time).total_seconds()
            self._response_times.append(response_time)
            response_data['response_time'] = response_time
            response_data['conversation_length'] = len(conversation['messages'])
            return response_data
            
        except Exception as e:
            await self.logger.error(f"Error in send_message for user {user_id}: {e}")
            return {
                'response': "I'm experiencing some technical difficulties. Please try again in a moment.",
                'confidence': 0.0,
                'error': str(e),
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the chat service."""
        if not self._response_times:
            return {'avg_response_time': 0.0, 'total_conversations': 0}
        return {
            'avg_response_time': sum(self._response_times) / len(self._response_times),
            'min_response_time': min(self._response_times),
            'max_response_time': max(self._response_times),
            'total_conversations': len(self._conversation_cache),
            'active_conversations': len([c for c in self._conversation_cache.values() if (datetime.now() - c.get('last_updated', datetime.now())).total_seconds() < 3600])
        }

    async def perform_action(self, *args, **kwargs):
        return await self.run_with_notification(self._perform_action_impl, *args, **kwargs)

    async def _perform_action_impl(self, *args, **kwargs):
        pass