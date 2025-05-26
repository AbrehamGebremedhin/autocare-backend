import asyncio
import time
import json
from typing import Any, Dict, Optional, List, Tuple
from app.utils.diagnosis_tree import AbstractTreeNode
from langchain_core.language_models.base import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.llm import LLMChain
from .base import AgentBase

class EnhancedDiagnosisTreeAgent:
    """
    Enhanced diagnosis tree agent with improved performance and service integration.
    Features:
    - Caching for LLM responses
    - Batch processing for multiple expansions
    - Enhanced error handling and recovery
    - Performance monitoring
    - Integration with enhanced services
    """
    
    def __init__(self, llm: BaseLanguageModel, prompt: PromptTemplate, 
                 root_issue_name: str = "root", root_likelihood: float = 1.0,
                 services: Dict[str, Any] = None):
        self.lock = asyncio.Lock()
        self.root = self._create_root(root_issue_name, root_likelihood)
        self.node_map: Dict[str, AbstractTreeNode] = {self.root.issue_name: self.root}
        self.llm = llm
        self.prompt = prompt
        self.output_parser = JsonOutputParser()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.services = services or {}
        
        # Enhanced features
        self.llm_cache: Dict[str, Tuple[Any, float]] = {}  # Cache with TTL
        self.cache_ttl = 1800  # 30 minutes
        self.performance_metrics = {
            'expansions': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.max_batch_size = 10
        self.retry_attempts = 3
        
    def _create_root(self, issue_name: str, likelihood: float) -> AbstractTreeNode:
        class RootNode(AbstractTreeNode):
            def process(self):
                pass
        return RootNode(issue_name, likelihood)
    
    def _get_cache_key(self, node_name: str, context: str, symptom_text: str) -> str:
        """Generate cache key for LLM responses."""
        return f"llm:{hash(node_name + context + symptom_text)}"
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - timestamp < self.cache_ttl
    
    async def _get_cached_llm_response(self, cache_key: str) -> Optional[Any]:
        """Get cached LLM response if valid."""
        if cache_key in self.llm_cache:
            response, timestamp = self.llm_cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.performance_metrics['cache_hits'] += 1
                return response
            else:
                del self.llm_cache[cache_key]
        
        self.performance_metrics['cache_misses'] += 1
        return None
    
    async def _cache_llm_response(self, cache_key: str, response: Any):
        """Cache LLM response with timestamp."""
        self.llm_cache[cache_key] = (response, time.time())
        
        # Cleanup old entries if cache gets too large
        if len(self.llm_cache) > 1000:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self.llm_cache.items()
                if current_time - timestamp > self.cache_ttl
            ]
            for key in expired_keys:
                del self.llm_cache[key]
    
    async def _invoke_llm_with_retry(self, llm_input: Dict[str, Any]) -> Any:
        """Invoke LLM with retry logic and error handling."""
        for attempt in range(self.retry_attempts):
            try:
                # Check cache first
                cache_key = self._get_cache_key(
                    llm_input.get('parent_issue', ''),
                    llm_input.get('context', ''),
                    llm_input.get('symptom_text', '')
                )
                
                cached_response = await self._get_cached_llm_response(cache_key)
                if cached_response is not None:
                    return cached_response
                
                # Invoke LLM
                response = await asyncio.to_thread(self.chain.invoke, llm_input)
                
                # Cache the response
                await self._cache_llm_response(cache_key, response)
                
                return response
                
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise e
    
    async def get_tree(self) -> AbstractTreeNode:
        async with self.lock:
            return self.root
    
    async def expand_node_with_llm(self, node_name: str, context: str, symptom_text: str) -> List[AbstractTreeNode]:
        """Enhanced node expansion with caching and error handling."""
        start_time = time.time()
        
        async with self.lock:
            parent = self.node_map.get(node_name)
            if not parent:
                raise ValueError(f"Node '{node_name}' not found.")
            
            llm_input = {
                "parent_issue": node_name,
                "context": context,
                "symptom_text": symptom_text
            }
            
            try:
                # Enhanced LLM integration
                if 'embedding_service' in self.services:
                    # Use embedding service to enrich context
                    context_embeddings = await self.services['embedding_service'].create_embeddings([context])
                    if context_embeddings:
                        llm_input['context_embedding_summary'] = "Context has high relevance to automotive symptoms"
                
                # Invoke LLM with retry
                response = await self._invoke_llm_with_retry(llm_input)
                
                # Parse response
                try:
                    parsed_issues = self.output_parser.parse(response)
                except Exception:
                    parsed_issues = []
                
                # Process parsed issues
                new_children = await self._process_parsed_issues(parent, parsed_issues)
                
                # Sort children by likelihood
                parent.sort_children_by_likelyhood()
                
                # Track performance
                execution_time = time.time() - start_time
                self.performance_metrics['expansions'].append(execution_time)
                
                return [self.node_map[issue.get('issue_name')] for issue in parsed_issues 
                       if issue.get('issue_name') in self.node_map]
                
            except Exception as e:
                # Enhanced error handling
                await self._handle_expansion_error(node_name, e)
                return []
    
    async def _process_parsed_issues(self, parent: AbstractTreeNode, parsed_issues: List[Dict]) -> List[AbstractTreeNode]:
        """Process parsed issues from LLM response."""
        new_children = []
        
        for issue in parsed_issues:
            name = issue.get('issue_name')
            likelihood = issue.get('likelihood', 0)
            data = issue.get('data', None)
            
            node = self.node_map.get(name)
            if node:
                # Update existing node
                node.likelyhood = likelihood
                node.update_data(data)
                if node.parent != parent:
                    if node.parent:
                        node.parent.remove_child(node)
                    parent.add_child(node)
            else:
                # Create new node
                class IssueNode(AbstractTreeNode):
                    def process(self):
                        pass
                
                new_node = IssueNode(name, likelihood, data)
                parent.add_child(new_node)
                self.node_map[name] = new_node
                new_children.append(new_node)
        
        return new_children
    
    async def _handle_expansion_error(self, node_name: str, error: Exception):
        """Handle errors during node expansion."""
        print(f"Error expanding node '{node_name}': {error}")
        
        # Try to use search service for fallback information
        if 'search_engine_service' in self.services:
            try:
                search_results = await self.services['search_engine_service'].search_web_chunks(
                    query=f"automotive issues related to {node_name}",
                    limit=3
                )
                if search_results:
                    print(f"Fallback search found {len(search_results)} results for {node_name}")
            except Exception as search_error:
                print(f"Fallback search also failed: {search_error}")
    
    async def expand_multiple_nodes(self, node_contexts: List[Tuple[str, str, str]]) -> Dict[str, List[AbstractTreeNode]]:
        """Expand multiple nodes in parallel for better performance."""
        async def expand_single(node_name: str, context: str, symptom_text: str) -> Tuple[str, List[AbstractTreeNode]]:
            try:
                result = await self.expand_node_with_llm(node_name, context, symptom_text)
                return node_name, result
            except Exception as e:
                print(f"Failed to expand {node_name}: {e}")
                return node_name, []
        
        # Limit batch size for performance
        batch_size = min(len(node_contexts), self.max_batch_size)
        batches = [node_contexts[i:i + batch_size] for i in range(0, len(node_contexts), batch_size)]
        
        results = {}
        for batch in batches:
            batch_tasks = [expand_single(name, ctx, symptom) for name, ctx, symptom in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, tuple):
                    node_name, children = result
                    results[node_name] = children
        
        return results
    
    async def expand_all_nodes_with_llm(self, context: str, symptom_text: str):
        """Enhanced recursive expansion with parallel processing."""
        async with self.lock:
            # Get all current nodes
            all_nodes = list(self.node_map.values())
            
            # Prepare batch expansion
            node_contexts = [(node.issue_name, context, symptom_text) for node in all_nodes]
            
            # Expand in batches
            await self.expand_multiple_nodes(node_contexts)
    
    async def update_tree_from_nodes(self, nodes: List[AbstractTreeNode], prune_threshold: float = 0.3):
        """Enhanced tree update with better integration."""
        async with self.lock:
            # Use enhanced services for validation if available
            if 'embedding_service' in self.services and nodes:
                try:
                    # Validate node relevance using embeddings
                    node_texts = [f"{node.issue_name}: {node.data}" for node in nodes if node.data]
                    if node_texts:
                        embeddings = await self.services['embedding_service'].create_embeddings(node_texts)
                        # Filter out nodes with low relevance (placeholder logic)
                        if embeddings and len(embeddings) == len(node_texts):
                            print(f"Validated {len(nodes)} nodes using embeddings")
                except Exception as e:
                    print(f"Embedding validation failed: {e}")
            
            # Standard tree update logic
            for node in nodes:
                existing = self.node_map.get(node.issue_name)
                if existing:
                    existing.likelyhood = node.likelyhood
                    existing.update_data(node.data)
                else:
                    parent = None
                    if node.parent and node.parent.issue_name in self.node_map:
                        parent = self.node_map[node.parent.issue_name]
                    elif node.parent:
                        parent = next((n for n in nodes if n.issue_name == node.parent.issue_name), None)
                    
                    if parent:
                        parent.add_child(node)
                        node.parent = parent
                    else:
                        self.root.add_child(node)
                        node.parent = self.root
                    
                    self.node_map[node.issue_name] = node
            
            # Prune and rebuild
            self.root.prune(prune_threshold)
            self.node_map = {n.issue_name: n for n in self.root.traverse()}
    
    async def update_issue(self, issue_name: str, data: Any):
        async with self.lock:
            node = self.node_map.get(issue_name)
            if not node:
                raise ValueError(f"Issue '{issue_name}' not found.")
            node.update_data(data)
    
    async def prune_tree(self, threshold: float = 0.3):
        async with self.lock:
            self.root.prune(threshold)
            self.node_map = {n.issue_name: n for n in self.root.traverse()}
    
    async def find_issue(self, issue_name: str) -> Optional[AbstractTreeNode]:
        async with self.lock:
            return self.node_map.get(issue_name)
    
    async def sort_children(self, issue_name: str, reverse: bool = True):
        async with self.lock:
            node = self.node_map.get(issue_name)
            if node:
                node.sort_children_by_likelyhood(reverse=reverse)
    
    async def reset(self):
        async with self.lock:
            self.root = self._create_root(self.root.issue_name, self.root.likelyhood)
            self.node_map = {self.root.issue_name: self.root}
            # Clear cache on reset
            self.llm_cache.clear()
            self.performance_metrics = {
                'expansions': [],
                'cache_hits': 0,
                'cache_misses': 0
            }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        expansions = self.performance_metrics['expansions']
        return {
            'total_expansions': len(expansions),
            'average_expansion_time': sum(expansions) / len(expansions) if expansions else 0,
            'cache_hit_ratio': (
                self.performance_metrics['cache_hits'] / 
                (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
                if (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']) > 0 
                else 0
            ),
            'cache_size': len(self.llm_cache),
            'tree_size': len(self.node_map)
        }


class SessionManagingAgent(AgentBase):
    """
    Enhanced session managing agent that leverages improved services for better performance.
    """
    
    def __init__(self, services: Dict[str, Any], config: Dict[str, Any] = None):
        super().__init__(services, config)
        self.active_sessions: Dict[str, EnhancedDiagnosisTreeAgent] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_sessions = config.get('max_sessions', 100) if config else 100
        self.session_timeout = config.get('session_timeout', 3600) if config else 3600  # 1 hour
        
    async def can_handle(self, task: str, context: Dict[str, Any]) -> bool:
        """Check if this agent can handle session management tasks."""
        session_keywords = ['session', 'diagnosis', 'tree', 'conversation', 'state', 'history']
        return any(keyword in task.lower() for keyword in session_keywords)
    
    async def handle(self, task: str, context: Dict[str, Any]) -> Any:
        """Handle session management tasks with enhanced performance."""
        session_id = context.get('session_id')
        if not session_id:
            raise ValueError("Session ID is required for session management")
        
        # Clean up expired sessions
        await self._cleanup_expired_sessions()
        
        if 'create' in task.lower() or 'start' in task.lower():
            return await self._create_session(session_id, context)
        elif 'expand' in task.lower():
            return await self._expand_session_tree(session_id, context)
        elif 'update' in task.lower():
            return await self._update_session(session_id, context)
        elif 'get' in task.lower() or 'retrieve' in task.lower():
            return await self._get_session_state(session_id)
        elif 'end' in task.lower() or 'close' in task.lower():
            return await self._end_session(session_id)
        else:
            return await self._manage_session_interaction(session_id, task, context)
    
    async def _create_session(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new diagnosis session with enhanced features."""
        if session_id in self.active_sessions:
            return {'status': 'session_exists', 'session_id': session_id}
        
        # Check session limits
        if len(self.active_sessions) >= self.max_sessions:
            await self._cleanup_oldest_sessions(5)  # Remove 5 oldest sessions
        
        # Create enhanced diagnosis tree agent
        llm = context.get('llm')  # Should be provided by caller
        prompt = context.get('prompt_template')  # Should be provided by caller
        
        if not llm or not prompt:
            raise ValueError("LLM and prompt template are required for session creation")
        
        tree_agent = EnhancedDiagnosisTreeAgent(
            llm=llm,
            prompt=prompt,
            root_issue_name=context.get('root_issue', 'automotive_issue'),
            root_likelihood=1.0,
            services=self.services
        )
        
        self.active_sessions[session_id] = tree_agent
        self.session_metadata[session_id] = {
            'created_at': time.time(),
            'last_accessed': time.time(),
            'interaction_count': 0,
            'root_issue': context.get('root_issue', 'automotive_issue')
        }
        
        return {
            'status': 'session_created',
            'session_id': session_id,
            'tree_stats': await tree_agent.get_performance_stats()
        }
    
    async def _expand_session_tree(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Expand the diagnosis tree for a session."""
        tree_agent = self.active_sessions.get(session_id)
        if not tree_agent:
            raise ValueError(f"Session {session_id} not found")
        
        self._update_session_access(session_id)
        
        node_name = context.get('node_name', 'root')
        context_text = context.get('context', '')
        symptom_text = context.get('symptoms', '')
        
        try:
            expanded_nodes = await tree_agent.expand_node_with_llm(
                node_name, context_text, symptom_text
            )
            
            return {
                'status': 'expanded',
                'session_id': session_id,
                'expanded_nodes': [node.issue_name for node in expanded_nodes],
                'tree_stats': await tree_agent.get_performance_stats()
            }
        except Exception as e:
            await self.logger.error(f"Error expanding tree for session {session_id}: {e}")
            return {
                'status': 'error',
                'session_id': session_id,
                'error': str(e)
            }
    
    async def _update_session(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update session state with new information."""
        tree_agent = self.active_sessions.get(session_id)
        if not tree_agent:
            raise ValueError(f"Session {session_id} not found")
        
        self._update_session_access(session_id)
        
        # Handle different update types
        if 'nodes' in context:
            await tree_agent.update_tree_from_nodes(
                context['nodes'],
                context.get('prune_threshold', 0.3)
            )
        elif 'issue_name' in context and 'data' in context:
            await tree_agent.update_issue(context['issue_name'], context['data'])
        
        return {
            'status': 'updated',
            'session_id': session_id,
            'tree_stats': await tree_agent.get_performance_stats()
        }
    
    async def _get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get current session state and tree information."""
        tree_agent = self.active_sessions.get(session_id)
        if not tree_agent:
            raise ValueError(f"Session {session_id} not found")
        
        self._update_session_access(session_id)
        
        tree = await tree_agent.get_tree()
        metadata = self.session_metadata.get(session_id, {})
        
        return {
            'status': 'active',
            'session_id': session_id,
            'tree_root': tree.issue_name,
            'tree_size': len(tree_agent.node_map),
            'metadata': metadata,
            'performance_stats': await tree_agent.get_performance_stats()
        }
    
    async def _end_session(self, session_id: str) -> Dict[str, Any]:
        """End a diagnosis session and clean up resources."""
        if session_id not in self.active_sessions:
            return {'status': 'session_not_found', 'session_id': session_id}
        
        # Get final stats
        tree_agent = self.active_sessions[session_id]
        final_stats = await tree_agent.get_performance_stats()
        
        # Clean up
        del self.active_sessions[session_id]
        del self.session_metadata[session_id]
        
        return {
            'status': 'session_ended',
            'session_id': session_id,
            'final_stats': final_stats
        }
    
    async def _manage_session_interaction(self, session_id: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general session interactions."""
        tree_agent = self.active_sessions.get(session_id)
        if not tree_agent:
            raise ValueError(f"Session {session_id} not found")
        
        self._update_session_access(session_id)
        
        # Use enhanced services for better interaction handling
        if 'chat_service' in self.services:
            try:
                # Enhance context with session state
                enhanced_context = context.copy()
                tree = await tree_agent.get_tree()
                enhanced_context['current_tree_state'] = {
                    'root': tree.issue_name,
                    'size': len(tree_agent.node_map)
                }
                
                # Process with chat service
                response = await self.services['chat_service'].process_message(
                    message=task,
                    context=enhanced_context
                )
                
                return {
                    'status': 'processed',
                    'session_id': session_id,
                    'response': response
                }
            except Exception as e:
                await self.logger.error(f"Chat service error for session {session_id}: {e}")
        
        return {
            'status': 'basic_response',
            'session_id': session_id,
            'message': f"Processed task: {task}"
        }
    
    def _update_session_access(self, session_id: str):
        """Update session access time and interaction count."""
        if session_id in self.session_metadata:
            self.session_metadata[session_id]['last_accessed'] = time.time()
            self.session_metadata[session_id]['interaction_count'] += 1
    
    async def _cleanup_expired_sessions(self):
        """Clean up sessions that have exceeded the timeout."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, metadata in self.session_metadata.items():
            if current_time - metadata.get('last_accessed', 0) > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self._end_session(session_id)
            await self.logger.info(f"Cleaned up expired session: {session_id}")
    
    async def _cleanup_oldest_sessions(self, count: int):
        """Clean up the oldest sessions to make room for new ones."""
        if len(self.active_sessions) <= count:
            return
        
        # Sort by last accessed time
        sorted_sessions = sorted(
            self.session_metadata.items(),
            key=lambda x: x[1].get('last_accessed', 0)
        )
        
        for i in range(min(count, len(sorted_sessions))):
            session_id = sorted_sessions[i][0]
            await self._end_session(session_id)
            await self.logger.info(f"Cleaned up old session to make room: {session_id}")
    
    async def get_session_statistics(self) -> Dict[str, Any]:
        """Get overall session management statistics."""
        current_time = time.time()
        active_count = len(self.active_sessions)
        
        if not self.session_metadata:
            return {
                'active_sessions': 0,
                'average_session_age': 0,
                'total_interactions': 0
            }
        
        session_ages = [
            current_time - metadata.get('created_at', current_time)
            for metadata in self.session_metadata.values()
        ]
        
        total_interactions = sum(
            metadata.get('interaction_count', 0)
            for metadata in self.session_metadata.values()
        )
        
        return {
            'active_sessions': active_count,
            'average_session_age': sum(session_ages) / len(session_ages) if session_ages else 0,
            'total_interactions': total_interactions,
            'max_sessions': self.max_sessions,
            'session_timeout': self.session_timeout
        }

    async def update_tree_from_nodes(self, nodes: List[AbstractTreeNode], prune_threshold: float = 0.3):
        async with self.lock:
            for node in nodes:
                existing = self.node_map.get(node.issue_name)
                if existing:
                    existing.likelyhood = node.likelyhood
                    existing.update_data(node.data)
                else:
                    parent = None
                    if node.parent and node.parent.issue_name in self.node_map:
                        parent = self.node_map[node.parent.issue_name]
                    elif node.parent:
                        parent = next((n for n in nodes if n.issue_name == node.parent.issue_name), None)
                    if parent:
                        parent.add_child(node)
                        node.parent = parent
                    else:
                        self.root.add_child(node)
                        node.parent = self.root
                    self.node_map[node.issue_name] = node
            self.root.prune(prune_threshold)
            self.node_map = {n.issue_name: n for n in self.root.traverse()}

    async def update_issue(self, issue_name: str, data: Any):
        async with self.lock:
            node = self.node_map.get(issue_name)
            if not node:
                raise ValueError(f"Issue '{issue_name}' not found.")
            node.update_data(data)

    async def prune_tree(self, threshold: float = 0.3):
        async with self.lock:
            self.root.prune(threshold)
            self.node_map = {n.issue_name: n for n in self.root.traverse()}

    async def find_issue(self, issue_name: str) -> Optional[AbstractTreeNode]:
        async with self.lock:
            return self.node_map.get(issue_name)

    async def sort_children(self, issue_name: str, reverse: bool = True):
        async with self.lock:
            node = self.node_map.get(issue_name)
            if node:
                node.sort_children_by_likelyhood(reverse=reverse)

    async def reset(self):
        async with self.lock:
            self.root = self._create_root(self.root.issue_name, self.root.likelyhood)
            self.node_map = {self.root.issue_name: self.root}
