from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from app.services.base_service import BaseService
from app.agents.orchestrator_agent import OrchestratorAgent
from collections import deque
from uuid import uuid4
import logging
from app.utils.diagnosis_tree import DiagnosisTreeNode
from app.core.interfaces import IWebSocketManager
from app.utils.message_types import MessageSource

class ChatService(BaseService):
    """
    ChatService with session management and direct message passing to OrchestratorAgent.
    """
    def __init__(
        self,
        websocket_manager: IWebSocketManager = None,
        orchestrator: Optional[OrchestratorAgent] = None,
        logger: Optional[logging.Logger] = None,
        conversation_cache: Optional[dict] = None,
        response_times: Optional[deque] = None,
        max_conversation_length: int = 20,
        conversation_ttl: int = 3600,
    ):
        super().__init__(websocket_manager=websocket_manager)
        self.orchestrator = orchestrator or OrchestratorAgent()
        self._conversation_cache = conversation_cache or {}  # Cache for active conversations
        self._max_conversation_length = max_conversation_length  # Maximum messages to keep in memory
        self._conversation_ttl = conversation_ttl  # 1 hour TTL for conversations
        self._response_times = response_times or deque(maxlen=100)  # Track last 100 response times
        self.logger = logger or logging.getLogger(__name__)

    def _get_conversation(self, user_id: str, session: Optional[Dict] = None) -> Dict:
        now = datetime.now()
        from app.schemas.Chat_Session import ChatSession
        
        # Check cache first, even if session is provided (cache might have updates)
        if user_id in self._conversation_cache:
            conversation = self._conversation_cache[user_id]
            last_updated = conversation.get('last_updated', now)
            
            # Handle both datetime objects and ISO strings for last_updated
            if isinstance(last_updated, str):
                try:
                    last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                except:
                    last_updated = now  # Fallback if parsing fails
            
            if (now - last_updated).total_seconds() < self._conversation_ttl:
                if len(conversation['messages']) > self._max_conversation_length:
                    conversation['messages'] = conversation['messages'][-self._max_conversation_length:]
                return conversation
            else:
                pass  # Cache expired, will use database session if provided
        
        if session:
            # Use the provided session (from DB)
            if 'last_updated' not in session:
                session['last_updated'] = now.isoformat()
            if len(session['messages']) > self._max_conversation_length:
                session['messages'] = session['messages'][-self._max_conversation_length:]
            # --- Ensure diagnosis_tree is a DiagnosisTreeNode ---
            # The diagnosis_tree is stored at the top level of session, not in context
            tree = session.get('diagnosis_tree')
            ctx = session.get('context', {})
            if not isinstance(ctx, dict):
                ctx = {}
            
            if tree and isinstance(tree, (dict, list)):
                # If it's a list, take the first element (legacy), else use as dict
                tree_data = tree[0] if isinstance(tree, list) and tree else tree
                try:
                    # Store the deserialized tree in context for easy access by orchestrator
                    ctx['diagnosis_tree'] = ChatSession.deserialize_diagnosis_tree(tree_data)
                except Exception as e:
                    # fallback: create new root
                    from app.utils.diagnosis_tree import DiagnosisTreeNode
                    ctx['diagnosis_tree'] = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
            elif tree and isinstance(tree, str):
                # Handle case where tree is stored as JSON string
                try:
                    import json
                    tree_data = json.loads(tree)
                    ctx['diagnosis_tree'] = ChatSession.deserialize_diagnosis_tree(tree_data)
                except Exception as e:
                    # fallback: create new root
                    from app.utils.diagnosis_tree import DiagnosisTreeNode
                    ctx['diagnosis_tree'] = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
            elif not ctx.get('diagnosis_tree'):
                # Create a new root if no tree exists
                from app.utils.diagnosis_tree import DiagnosisTreeNode
                ctx['diagnosis_tree'] = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
            else:
                # Use existing diagnosis tree from context
                pass
            
            session['context'] = ctx
            # Ensure title exists, set default if missing
            if 'title' not in session or not session['title']:
                session['title'] = 'New Chat Session'
            self._conversation_cache[user_id] = session
            return session
            
        # Create new conversation
        session_id = str(uuid4())
        # Create a diagnosis tree instance for this session
        diagnosis_tree = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
        conversation = {
            'id': session_id,
            'user_id': user_id,
            'title': 'New Chat Session',  # Default title, will be updated after first interaction
            'messages': [],
            'created_at': now,
            'updated_at': now,
            'context': {'diagnosis_tree': diagnosis_tree},
        }
        self._conversation_cache[user_id] = conversation
        return conversation

    @BaseService.cache_result(ttl_seconds=300)
    async def send_message(self, user_id: str, message: str, context: Optional[Dict] = None, session: Optional[Dict] = None, websocket=None) -> Dict[str, Any]:
        """
        Send a message from a user and return the response from the OrchestratorAgent.
        Args:
            user_id: Unique identifier for the user
            message: User's message
            context: Optional context information (car details, symptoms, etc.)
            session: Optional loaded session from DB
            websocket: Optional WebSocket for direct messaging
        Returns:
            Dict containing response, confidence, sources, and metadata
        """
        start_time = datetime.now()
        try:
            await self._rate_limit()  # Apply rate limiting
            
            conversation = self._get_conversation(user_id, session=session)
            is_initial = len(conversation['messages']) == 0
            
            # Initialize session context agent for tracking original issue
            if is_initial:
                from app.agents.session_context_agent import SessionContextAgent, OriginalIssueContext
                context_agent = SessionContextAgent(
                    car_id=self._get_car_id_from_context(context),
                    car_make=self._get_car_make_from_context(context),
                    car_model=self._get_car_model_from_context(context),
                    car_year=self._get_car_year_from_context(context)
                )
                # Extract and store the original issue context using LLM
                original_context = await context_agent.extract_original_issue(
                    message, 
                    conversation['id'],
                    self._get_car_make_from_context(context) or "",
                    self._get_car_model_from_context(context) or "",
                    self._get_car_year_from_context(context) or ""
                )
                # Store it in the conversation context for persistence
                conversation['context']['original_issue'] = original_context.to_dict()
            else:
                # For non-initial messages, check if they're relevant to the original issue
                if 'original_issue' in conversation['context']:
                    from app.agents.session_context_agent import SessionContextAgent, OriginalIssueContext
                    context_agent = SessionContextAgent(
                        car_id=self._get_car_id_from_context(context),
                        car_make=self._get_car_make_from_context(context),
                        car_model=self._get_car_model_from_context(context),
                        car_year=self._get_car_year_from_context(context)
                    )
                    # Restore the original issue context
                    try:
                        original_issue_data = conversation['context']['original_issue']
                        if original_issue_data is None:
                            self.logger.warning(f"Original issue data is None for session {conversation['id']}")
                            # Continue without context enforcement
                        else:
                            original_context = OriginalIssueContext.from_dict(original_issue_data)
                            if original_context is None:
                                self.logger.warning(f"Failed to create OriginalIssueContext from data for session {conversation['id']}")
                                # Continue without context enforcement
                            else:
                                context_agent.original_issue_contexts[conversation['id']] = original_context
                                
                                # Check if current message is relevant using LLM
                                is_relevant = await context_agent.is_message_relevant(message, conversation['id'])
                                
                                if not is_relevant:
                                    self.logger.warning(f"User message may be off-topic from original {original_context.issue_category} issue: {message[:100]}")
                                    
                                    # Return a controlled response for off-topic messages
                                    return {
                                        'response': f"That sounds like it might be a different automotive issue. Since this session is focused on your {original_context.issue_category} problem ({original_context.primary_issue}), I'd recommend starting a new session to properly address that concern. Would you like to continue discussing the {original_context.issue_category} issue, or shall we focus on getting that resolved first?",
                                        'confidence': 0.8,
                                        'user_id': user_id,
                                        'timestamp': datetime.now().isoformat(),
                                        'session_focus_maintained': True,
                                        'original_issue': original_context.primary_issue
                                    }
                                else:
                                    # Message is relevant - let specialized agents handle symptom extraction
                                    self.logger.info(f"Session {conversation['id']}: Message is relevant to {original_context.issue_category} issue")
                                    
                                    # Add context guidance for the orchestrator
                                    if context is None:
                                        context = {}
                                    context['session_context'] = context_agent.get_context_reminder(conversation['id'])
                            
                    except Exception as e:
                        import traceback
                        self.logger.error(f"Failed to restore original issue context: {e}")
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                        # Continue without context enforcement if restoration fails
            # Ensure conversation['context'] is a dict
            if not isinstance(conversation.get('context'), dict):
                import json as _json
                try:
                    conversation['context'] = _json.loads(conversation['context'])
                except Exception:
                    conversation['context'] = {}
            # Add user message to conversation
            # Create a clean context for message storage (exclude diagnosis_tree to avoid serialization issues)
            message_context = {k: v for k, v in (context or {}).items() if k != 'diagnosis_tree'}
            conversation['messages'].append({
                'role': 'user',
                'content': message,
                'timestamp': start_time.isoformat(),
                'context': message_context,
                'is_initial': is_initial
            })
            # Mark initial message in context for orchestrator
            if context is None:
                context = {}
            if is_initial:
                context = dict(context)
                context['is_initial_message'] = True
            # Always pass the session's diagnosis tree in context
            context['diagnosis_tree'] = conversation['context'].get('diagnosis_tree')
            
            # Pass conversation history for context-aware responses
            context['conversation_history'] = conversation['messages']
            
            # Send progress message
            if websocket:
                try:
                    await self.send_ws_progress(websocket, "Processing user message", MessageSource.CHAT_SERVICE, 0.1, session_id=conversation['id'])
                except Exception as ws_error:
                    self.logger.error(f"WebSocket send_ws_progress error: {ws_error}")
            # Send to orchestrator agent
            response_data = await self.orchestrator.route_request(
                message, 
                user_id, 
                context, 
                websocket=websocket, 
                session_id=conversation['id']
            )
            
            # Ensure response_data is not None
            if response_data is None:
                self.logger.warning("Orchestrator returned None response_data")
                response_data = {
                    'response': "I apologize, but I couldn't process your request at this time. Please try again.",
                    'confidence': 0.0,
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Update the diagnosis tree in the conversation context if returned by orchestrator
            if response_data and 'diagnosis_tree' in response_data:
                conversation['context']['diagnosis_tree'] = response_data['diagnosis_tree']
                
                # Update the original issue context with new diagnosis findings
                if 'original_issue' in conversation['context']:
                    # Session context agent doesn't handle diagnosis updates - it only manages session focus
                    # The diagnosis tree updates are handled by the orchestrator and stored in conversation context
                    self.logger.info("Diagnosis tree updated in conversation context")
                
                # Only update session title if it's still the default title
                current_title = conversation.get('title', 'New Chat Session')
                if current_title == 'New Chat Session':
                    from app.schemas.Chat_Session import ChatSession
                    new_title = ChatSession.generate_session_title(
                        conversation['context']['diagnosis_tree'], 
                        conversation['messages']
                    )
                    conversation['title'] = new_title
                else:
                    # Session title unchanged - not default title
                    pass
            else:
                # No diagnosis tree returned by orchestrator - likely a simple response
                # Still generate a title for simple chat sessions
                current_title = conversation.get('title', 'New Chat Session')
                if current_title == 'New Chat Session':
                    # Generate title for simple responses based on conversation content
                    simple_title = await self.generate_simple_session_title(conversation['messages'])
                    conversation['title'] = simple_title
            
            # Add assistant response to conversation
            conversation['messages'].append({
                'role': 'assistant',
                'content': response_data.get('response', ''),  # Now this is the user_message
                'timestamp': datetime.now().isoformat(),
                'confidence': response_data.get('confidence', 0.0),
                'sources': response_data.get('sources', [])
            })
            # Convert datetime fields to isoformat before saving
            conversation['last_updated'] = datetime.now().isoformat()
            if isinstance(conversation.get('created_at'), datetime):
                conversation['created_at'] = conversation['created_at'].isoformat()
            self._conversation_cache[user_id] = conversation
            response_time = (datetime.now() - start_time).total_seconds()
            self._response_times.append(response_time)
            response_data['response_time'] = response_time
            response_data['conversation_length'] = len(conversation['messages'])
            # Send result message
            if websocket:
                try:
                    await self.send_ws_result(websocket, "Response ready", MessageSource.CHAT_SERVICE, session_id=conversation['id'], details=response_data)
                except Exception as ws_error:
                    self.logger.error(f"WebSocket send_ws_result error: {ws_error}")
            return response_data
        except Exception as e:
            self.logger.error(f"Error in send_message for user {user_id}: {e}")
            if websocket:
                try:
                    await self.send_ws_error(websocket, "Error processing message", MessageSource.CHAT_SERVICE, session_id=None, details={"error": str(e)})
                except Exception as ws_error:
                    self.logger.error(f"WebSocket send_ws_error error: {ws_error}")
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
        
    # Helper methods for context extraction
    def _get_car_id_from_context(self, context: Optional[Dict]) -> Optional[str]:
        """Extract car ID from conversation context."""
        if not context:
            return None
        if 'car' in context:
            return context['car'].get('id')
        if 'original_issue' in context:
            return context['original_issue'].get('car_id')
        return None
        
    def _get_car_make_from_context(self, context: Optional[Dict]) -> Optional[str]:
        """Extract car make from conversation context."""
        if not context:
            return None
        if 'car' in context:
            return context['car'].get('make')
        if 'original_issue' in context:
            return context['original_issue'].get('car_make')
        return None
        
    def _get_car_model_from_context(self, context: Optional[Dict]) -> Optional[str]:
        """Extract car model from conversation context."""
        if not context:
            return None
        if 'car' in context:
            return context['car'].get('model')
        if 'original_issue' in context:
            return context['original_issue'].get('car_model')
        return None
        
    def _get_car_year_from_context(self, context: Optional[Dict]) -> Optional[int]:
        """Extract car year from conversation context."""
        if not context:
            return None
        if 'car' in context:
            year = context['car'].get('year')
            return int(year) if year else None
        if 'original_issue' in context:
            year = context['original_issue'].get('car_year')
            return int(year) if year else None
        return None

    async def generate_simple_session_title(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a meaningful title for simple chat sessions based on conversation content.
        Uses LLM to create concise, descriptive titles for general automotive discussions.
        """
        try:
            if not messages:
                return "New Chat Session"
            
            # Get the first user message for context
            first_user_msg = next((msg for msg in messages if msg.get('role') == 'user'), None)
            if not first_user_msg:
                return "General Chat"
            
            first_content = first_user_msg.get('content', '')
            
            # If there are multiple messages, get recent conversation context
            conversation_summary = ""
            if len(messages) > 2:
                # Include last few messages for context
                recent_messages = messages[-4:]  # Last 4 messages
                for msg in recent_messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if isinstance(content, dict):
                        content = content.get('technical_diagnosis', str(content)[:100])
                    elif isinstance(content, str):
                        content = content[:100]
                    conversation_summary += f"{role}: {content}\n"
            
            # Use LLM to generate a concise title
            from app.services.llm_service import LLMService
            llm_service = LLMService()
            
            prompt = f"""
            Generate a short, descriptive title (maximum 6 words) for this automotive chat session.
            
            FIRST MESSAGE: "{first_content}"
            
            RECENT CONVERSATION:
            {conversation_summary or "No additional context"}
            
            The title should:
            - Be concise and descriptive
            - Capture the main topic or question
            - Be relevant to automotive content
            - Use simple, clear language
            
            Examples of good titles:
            - "Diesel vs Benzene Comparison"
            - "Engine Oil Change Guide"
            - "Brake Noise Troubleshooting"
            - "Transmission Fluid Questions"
            
            Return only the title, no explanations or quotes.
            """
            
            response = await llm_service.generate_response(prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                title = response.content.strip()
            elif isinstance(response, dict):
                title = response.get('content', str(response)).strip()
            else:
                title = str(response).strip()
            
            # Clean up the title
            title = title.strip('"\'')  # Remove quotes
            title = title[:50]  # Limit length
            
            # Fallback if LLM response is empty or invalid
            if not title or len(title) < 3:
                # Extract key words from first message as fallback
                words = first_content.split()[:6]  # First 6 words
                title = " ".join(words)
                if len(first_content) > 50:
                    title += "..."
            
            return title if title else "General Automotive Chat"
            
        except Exception as e:
            self.logger.error(f"Failed to generate simple session title: {e}")
            # Fallback to extracting from first message
            if messages:
                first_user_msg = next((msg for msg in messages if msg.get('role') == 'user'), None)
                if first_user_msg and first_user_msg.get('content'):
                    content = first_user_msg['content'][:40]
                    if len(first_user_msg['content']) > 40:
                        content += "..."
                    return content
            return "General Automotive Chat"