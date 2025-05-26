from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import json
from app.services.base_service import BaseService
from app.services.embedding_service import EmbeddingService
from app.services.search_engine_service import SearchEngineService
from app.services.query_builder_service import QueryBuilderService
from app.utils.logger import Logger
import httpx
from collections import deque

class ChatService(BaseService):
    """
    Enhanced ChatService with performance optimizations, caching, and improved conversation handling.
    """
    
    def __init__(self, 
                 embedding_service: Optional[EmbeddingService] = None,
                 search_service: Optional[SearchEngineService] = None,
                 query_builder: Optional[QueryBuilderService] = None,
                 logger: Optional[Logger] = None):
        """
        Initialize the ChatService with enhanced capabilities.
        Args:
            embedding_service: Service for generating embeddings
            search_service: Service for searching knowledge base
            query_builder: Service for building optimized queries
            logger: Optional logger instance
        """
        super().__init__()
        self.embedding_service = embedding_service or EmbeddingService()
        self.search_service = search_service or SearchEngineService()
        self.query_builder = query_builder or QueryBuilderService()
        self.logger = logger or Logger("ChatService")
        
        # Conversation management
        self._conversation_cache = {}  # Cache for active conversations
        self._max_conversation_length = 20  # Maximum messages to keep in memory
        self._conversation_ttl = 3600  # 1 hour TTL for conversations
        
        # Response optimization
        self._response_templates = {
            'greeting': "Hello! I'm here to help you with automotive questions. What can I assist you with today?",
            'clarification': "Could you provide more details about your {topic}? For example, what specific symptoms are you experiencing?",
            'no_results': "I couldn't find specific information about that. Could you rephrase your question or provide more context?",
            'error': "I'm experiencing some technical difficulties. Please try again in a moment."
        }
        
        # Performance tracking
        self._response_times = deque(maxlen=100)  # Track last 100 response times
        
    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self.embedding_service, 'cleanup'):
            await self.embedding_service.cleanup()
        if hasattr(self.search_service, 'cleanup'):
            await self.search_service.cleanup()
        if hasattr(self.query_builder, 'cleanup'):
            await self.query_builder.cleanup()

    @BaseService.cache_result(ttl_seconds=300)  # Cache responses for 5 minutes
    async def send_message(self, user_id: str, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Send a message from a user and return an enhanced response with context awareness.
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
            
            # Get or create conversation context
            conversation = self._get_conversation(user_id)
            
            # Add user message to conversation
            conversation['messages'].append({
                'role': 'user',
                'content': message,
                'timestamp': start_time.isoformat(),
                'context': context
            })
            
            # Process message with enhanced logic
            response_data = await self._process_message_enhanced(user_id, message, conversation, context)
            
            # Add assistant response to conversation
            conversation['messages'].append({
                'role': 'assistant',
                'content': response_data['response'],
                'timestamp': datetime.now().isoformat(),
                'confidence': response_data.get('confidence', 0.0),
                'sources': response_data.get('sources', [])
            })
            
            # Update conversation cache
            conversation['last_updated'] = datetime.now()
            self._conversation_cache[user_id] = conversation
            
            # Track performance
            response_time = (datetime.now() - start_time).total_seconds()
            self._response_times.append(response_time)
            
            # Add performance metadata
            response_data.update({
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'response_time': response_time,
                'conversation_length': len(conversation['messages'])
            })
            
            return response_data
            
        except Exception as e:
            await self.logger.error(f"Error in send_message for user {user_id}: {e}")
            return {
                'response': self._response_templates['error'],
                'confidence': 0.0,
                'error': str(e),
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }

    async def _process_message_enhanced(self, user_id: str, message: str, 
                                      conversation: Dict, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Enhanced message processing with parallel operations and context awareness.
        """
        try:
            # Extract intent and entities in parallel
            async def extract_intent():
                return await self._extract_intent(message, conversation)
            
            async def extract_entities():
                return await self._extract_automotive_entities(message, context)
            
            async def build_queries():
                return await self.query_builder.build_contextual_queries(message, context)
            
            # Execute tasks in parallel
            intent_task = asyncio.create_task(extract_intent())
            entities_task = asyncio.create_task(extract_entities())
            queries_task = asyncio.create_task(build_queries())
            
            intent, entities, queries = await asyncio.gather(
                intent_task, entities_task, queries_task, return_exceptions=True
            )
            
            # Handle any exceptions
            if isinstance(intent, Exception):
                intent = 'general'
            if isinstance(entities, Exception):
                entities = {}
            if isinstance(queries, Exception):
                queries = [message]
            
            # Search for relevant information
            search_results = await self._search_knowledge_base(queries, entities, context)
            
            # Generate response based on intent and results
            response = await self._generate_response(intent, entities, search_results, conversation)
            
            return {
                'response': response['text'],
                'confidence': response['confidence'],
                'sources': search_results.get('sources', []),
                'intent': intent,
                'entities': entities,
                'queries_used': queries[:3]  # Limit to first 3 queries for response
            }
            
        except Exception as e:
            await self.logger.error(f"Error in _process_message_enhanced: {e}")
            return {
                'response': self._response_templates['error'],
                'confidence': 0.0,
                'error': str(e)
            }

    async def _extract_intent(self, message: str, conversation: Dict) -> str:
        """Extract user intent from message with conversation context."""
        message_lower = message.lower()
        
        # Intent patterns
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'start']):
            return 'greeting'
        elif any(word in message_lower for word in ['problem', 'issue', 'trouble', 'broken', 'not working']):
            return 'diagnostic'
        elif any(word in message_lower for word in ['how to', 'repair', 'fix', 'replace', 'install']):
            return 'repair_instruction'
        elif any(word in message_lower for word in ['maintenance', 'service', 'schedule', 'when to']):
            return 'maintenance'
        elif any(word in message_lower for word in ['cost', 'price', 'expensive', 'cheap']):
            return 'cost_inquiry'
        elif any(word in message_lower for word in ['part', 'component', 'where to buy']):
            return 'parts_inquiry'
        else:
            return 'general'

    async def _extract_automotive_entities(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract automotive-specific entities from the message."""
        entities = {}
        message_lower = message.lower()
        
        # Vehicle parts/systems
        parts = ['engine', 'brake', 'transmission', 'battery', 'alternator', 'starter', 
                'radiator', 'carburetor', 'fuel pump', 'spark plug', 'oil filter', 
                'air filter', 'tire', 'wheel', 'suspension', 'exhaust', 'clutch']
        
        for part in parts:
            if part in message_lower:
                entities.setdefault('parts', []).append(part)
        
        # Symptoms
        symptoms = ['noise', 'leak', 'smoke', 'vibration', 'stall', 'overheat', 
                   'rough idle', 'hard start', 'no start', 'grinding', 'squealing']
        
        for symptom in symptoms:
            if symptom in message_lower:
                entities.setdefault('symptoms', []).append(symptom)
        
        # Add context information if available
        if context:
            entities['context'] = context
            
        return entities

    async def _search_knowledge_base(self, queries: List[str], entities: Dict, 
                                   context: Optional[Dict] = None) -> Dict[str, Any]:
        """Search knowledge base with optimized parallel queries."""
        try:
            # Limit concurrent searches to prevent overwhelming the system
            max_concurrent = min(3, len(queries))
            search_tasks = []
            
            for i in range(max_concurrent):
                if i < len(queries):
                    task = asyncio.create_task(
                        self.search_service.search_with_context(queries[i], context)
                    )
                    search_tasks.append(task)
            
            if not search_tasks:
                return {'results': [], 'sources': [], 'confidence': 0.0}
            
            # Execute searches in parallel
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and rank results
            combined_results = []
            sources = []
            
            for result in search_results:
                if not isinstance(result, Exception) and result:
                    combined_results.extend(result.get('results', []))
                    sources.extend(result.get('sources', []))
            
            # Remove duplicates and rank by relevance
            unique_results = self._deduplicate_results(combined_results)
            ranked_results = self._rank_results(unique_results, entities)
            
            return {
                'results': ranked_results[:5],  # Top 5 results
                'sources': sources[:3],  # Top 3 sources
                'confidence': self._calculate_confidence(ranked_results)
            }
            
        except Exception as e:
            await self.logger.error(f"Error in _search_knowledge_base: {e}")
            return {'results': [], 'sources': [], 'confidence': 0.0}

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content similarity."""
        if not results:
            return []
        
        unique_results = []
        seen_content = set()
        
        for result in results:
            content_hash = hash(result.get('content', '')[:200])  # Use first 200 chars for comparison
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results

    def _rank_results(self, results: List[Dict], entities: Dict) -> List[Dict]:
        """Rank results based on entity matching and relevance."""
        if not results:
            return []
        
        for result in results:
            score = result.get('score', 0.0)
            
            # Boost score for entity matches
            content_lower = result.get('content', '').lower()
            
            # Check for part matches
            for part in entities.get('parts', []):
                if part in content_lower:
                    score += 0.1
            
            # Check for symptom matches
            for symptom in entities.get('symptoms', []):
                if symptom in content_lower:
                    score += 0.15
            
            result['final_score'] = score
        
        # Sort by final score
        return sorted(results, key=lambda x: x.get('final_score', 0.0), reverse=True)

    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence score based on search results quality."""
        if not results:
            return 0.0
        
        # Base confidence on top result score and number of results
        top_score = results[0].get('final_score', 0.0) if results else 0.0
        result_count_factor = min(len(results) / 3.0, 1.0)  # Normalize to max of 1.0
        
        confidence = (top_score * 0.7) + (result_count_factor * 0.3)
        return min(confidence, 1.0)

    async def _generate_response(self, intent: str, entities: Dict, 
                               search_results: Dict, conversation: Dict) -> Dict[str, Any]:
        """Generate contextual response based on intent and search results."""
        try:
            results = search_results.get('results', [])
            confidence = search_results.get('confidence', 0.0)
            
            if not results or confidence < 0.2:
                # Low confidence or no results
                if entities.get('parts') or entities.get('symptoms'):
                    topic = ', '.join(entities.get('parts', []) + entities.get('symptoms', []))
                    response = self._response_templates['clarification'].format(topic=topic)
                else:
                    response = self._response_templates['no_results']
                return {'text': response, 'confidence': 0.1}
            
            # Generate response based on intent
            if intent == 'greeting':
                response = self._response_templates['greeting']
                if results:
                    response += f"\n\nI found some relevant information that might help: {results[0].get('content', '')[:200]}..."
                    
            elif intent == 'diagnostic':
                response = self._generate_diagnostic_response(entities, results)
                
            elif intent == 'repair_instruction':
                response = self._generate_repair_response(entities, results)
                
            elif intent == 'maintenance':
                response = self._generate_maintenance_response(entities, results)
                
            else:
                # General response
                response = self._generate_general_response(results)
            
            return {'text': response, 'confidence': confidence}
            
        except Exception as e:
            await self.logger.error(f"Error in _generate_response: {e}")
            return {'text': self._response_templates['error'], 'confidence': 0.0}

    def _generate_diagnostic_response(self, entities: Dict, results: List[Dict]) -> str:
        """Generate diagnostic-focused response."""
        parts = entities.get('parts', [])
        symptoms = entities.get('symptoms', [])
        
        response = "Based on your description, here's what I found:\n\n"
        
        if parts and symptoms:
            response += f"For {', '.join(parts)} issues with {', '.join(symptoms)}:\n"
        elif parts:
            response += f"Regarding {', '.join(parts)} problems:\n"
        elif symptoms:
            response += f"For the {', '.join(symptoms)} you're experiencing:\n"
        
        if results:
            response += f"{results[0].get('content', '')[:300]}..."
            if len(results) > 1:
                response += f"\n\nAdditional considerations: {results[1].get('content', '')[:200]}..."
        
        return response

    def _generate_repair_response(self, entities: Dict, results: List[Dict]) -> str:
        """Generate repair instruction response."""
        parts = entities.get('parts', [])
        
        response = "Here are the repair instructions I found:\n\n"
        
        if parts:
            response += f"For {', '.join(parts)} repair:\n"
        
        if results:
            # Look for step-by-step instructions
            for result in results[:2]:
                content = result.get('content', '')
                if any(word in content.lower() for word in ['step', 'procedure', 'remove', 'install']):
                    response += f"{content[:400]}...\n\n"
                    break
            else:
                response += f"{results[0].get('content', '')[:400]}..."
        
        return response

    def _generate_maintenance_response(self, entities: Dict, results: List[Dict]) -> str:
        """Generate maintenance-focused response."""
        response = "Here's the maintenance information I found:\n\n"
        
        if results:
            # Look for maintenance schedules and intervals
            for result in results[:2]:
                content = result.get('content', '')
                if any(word in content.lower() for word in ['schedule', 'interval', 'miles', 'months']):
                    response += f"{content[:300]}...\n\n"
                    break
            else:
                response += f"{results[0].get('content', '')[:300]}..."
        
        return response

    def _generate_general_response(self, results: List[Dict]) -> str:
        """Generate general response."""
        if not results:
            return self._response_templates['no_results']
        
        response = "Here's what I found that might help:\n\n"
        response += f"{results[0].get('content', '')[:400]}..."
        
        if len(results) > 1:
            response += f"\n\nYou might also find this helpful: {results[1].get('content', '')[:200]}..."
        
        return response

    def _get_conversation(self, user_id: str) -> Dict:
        """Get or create conversation context for user."""
        now = datetime.now()
        
        if user_id in self._conversation_cache:
            conversation = self._conversation_cache[user_id]
            
            # Check if conversation is still valid
            last_updated = conversation.get('last_updated', now)
            if (now - last_updated).total_seconds() < self._conversation_ttl:
                # Trim conversation if it's too long
                if len(conversation['messages']) > self._max_conversation_length:
                    conversation['messages'] = conversation['messages'][-self._max_conversation_length:]
                return conversation
        
        # Create new conversation
        return {
            'user_id': user_id,
            'messages': [],
            'created': now,
            'last_updated': now,
            'context': {}
        }

    @BaseService.cache_result(ttl_seconds=600)  # Cache for 10 minutes
    async def get_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve enhanced chat history for a user with performance optimizations.
        Args:
            user_id: User identifier
            limit: Maximum number of messages to return
        Returns:
            List of conversation messages with metadata
        """
        try:
            await self._rate_limit()
            
            conversation = self._get_conversation(user_id)
            messages = conversation.get('messages', [])
            
            # Apply limit and return most recent messages
            limited_messages = messages[-limit:] if len(messages) > limit else messages
            
            # Add performance metadata
            avg_response_time = sum(self._response_times) / len(self._response_times) if self._response_times else 0.0
            
            return {
                'messages': limited_messages,
                'total_messages': len(messages),
                'conversation_age': (datetime.now() - conversation.get('created', datetime.now())).total_seconds(),
                'avg_response_time': avg_response_time,
                'user_id': user_id
            }
            
        except Exception as e:
            await self.logger.error(f"Error retrieving history for user {user_id}: {e}")
            return {'messages': [], 'error': str(e), 'user_id': user_id}

    async def clear_conversation(self, user_id: str) -> bool:
        """Clear conversation history for a user."""
        try:
            if user_id in self._conversation_cache:
                del self._conversation_cache[user_id]
            return True
        except Exception as e:
            await self.logger.error(f"Error clearing conversation for user {user_id}: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the chat service."""
        if not self._response_times:
            return {'avg_response_time': 0.0, 'total_conversations': 0}
        
        return {
            'avg_response_time': sum(self._response_times) / len(self._response_times),
            'min_response_time': min(self._response_times),
            'max_response_time': max(self._response_times),
            'total_conversations': len(self._conversation_cache),
            'active_conversations': len([c for c in self._conversation_cache.values() 
                                       if (datetime.now() - c.get('last_updated', datetime.now())).total_seconds() < 3600])
        }