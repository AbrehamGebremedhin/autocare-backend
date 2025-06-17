from typing import Dict, Any, Optional
from datetime import datetime
from app.services.base_service import BaseService
from app.agents.orchestrator_agent import OrchestratorAgent

class ChatService(BaseService):
    """
    Simplified ChatService that directly passes messages to the OrchestratorAgent.
    """
    
    def __init__(self, 
                 websocket_manager=None):
        """
        Initialize the ChatService.
        Args:
            websocket_manager: Optional WebSocketManager for notifications.
        """
        super().__init__(websocket_manager=websocket_manager)
        self.orchestrator = OrchestratorAgent()
        self._conversation_cache = {}  # Cache for conversations

    @BaseService.cache_result(ttl_seconds=300)  # Cache responses for 5 minutes
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
            
            # Directly pass the message to the orchestrator agent
            response_data = await self.orchestrator.route_request(message, user_id, context)
            
            # Track performance
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Add performance metadata
            response_data['response_time'] = response_time
            
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
        return {
            'avg_response_time': 0.0,
            'total_conversations': len(self._conversation_cache),
            'active_conversations': len([c for c in self._conversation_cache.values() 
                                       if (datetime.now() - c.get('last_updated', datetime.now())).total_seconds() < 3600])
        }

    async def perform_action(self, *args, **kwargs):
        return await self.run_with_notification(self._perform_action_impl, *args, **kwargs)

    async def _perform_action_impl(self, *args, **kwargs):
        pass