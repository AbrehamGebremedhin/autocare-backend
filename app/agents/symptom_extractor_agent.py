import asyncio
import time
import json
from typing import Any, List, Optional, Dict, Tuple
from langchain_core.language_models.base import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.llm import LLMChain
from app.utils.logger import Logger
from .base import AgentBase

class SymptomExtractorAgent(AgentBase):
    """
    Enhanced symptom extractor agent that leverages improved services for better accuracy and performance.
    Features:
    - Advanced text processing with embedding service
    - Intelligent symptom extraction using search engine
    - Caching for repeated queries
    - Batch processing for multiple texts
    - Integration with enhanced chat service
    """
    
    def __init__(self, services: Dict[str, Any], config: Dict[str, Any] = None):
        super().__init__(services, config)
        self.symptom_patterns = self._load_symptom_patterns()
        self.automotive_keywords = self._load_automotive_keywords()
        self.confidence_threshold = config.get('confidence_threshold', 0.7) if config else 0.7
        self.max_batch_size = config.get('max_batch_size', 10) if config else 10
        self.cache_ttl = config.get('cache_ttl', 1800) if config else 1800  # 30 minutes
        
        # Performance tracking
        self.extraction_metrics = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'cache_hits': 0,
            'processing_times': []
        }
    
    def _load_symptom_patterns(self) -> List[str]:
        """Load predefined symptom patterns for automotive issues."""
        return [
            r'\b(?:engine|motor)\s+(?:knocking|rattling|stalling|overheating|misfiring)\b',
            r'\b(?:brake|brakes)\s+(?:squealing|grinding|soft|hard|vibrating)\b',
            r'\b(?:transmission|gearbox)\s+(?:slipping|rough|delayed|hard)\b',
            r'\b(?:steering)\s+(?:hard|loose|vibrating|pulling)\b',
            r'\b(?:suspension)\s+(?:bouncing|sagging|clunking)\b',
            r'\b(?:electrical|battery|alternator)\s+(?:dead|dim|flickering|not working)\b',
            r'\b(?:exhaust|muffler)\s+(?:loud|smoking|dragging)\b',
            r'\b(?:tire|tires|wheel|wheels)\s+(?:flat|worn|uneven|vibrating)\b',
            r'\b(?:oil|fluid)\s+(?:leak|low|dirty|burning)\b',
            r'\b(?:air conditioning|ac|heater)\s+(?:not working|weak|hot|cold)\b',
        ]
    
    def _load_automotive_keywords(self) -> List[str]:
        """Load automotive-specific keywords for context enhancement."""
        return [
            'vehicle', 'car', 'truck', 'automobile', 'engine', 'motor', 'transmission',
            'brake', 'steering', 'suspension', 'tire', 'wheel', 'exhaust', 'muffler',
            'battery', 'alternator', 'starter', 'radiator', 'coolant', 'oil', 'filter',
            'spark plug', 'fuel pump', 'carburetor', 'injector', 'sensor', 'belt',
            'hose', 'gasket', 'clutch', 'differential', 'axle', 'shock', 'strut'
        ]
    
    async def can_handle(self, task: str, context: Dict[str, Any]) -> bool:
        """Check if this agent can handle symptom extraction tasks."""
        symptom_keywords = [
            'symptom', 'extract', 'identify', 'analyze', 'problem', 'issue', 
            'complaint', 'description', 'text', 'parse'
        ]
        return any(keyword in task.lower() for keyword in symptom_keywords)
    
    async def handle(self, task: str, context: Dict[str, Any]) -> Any:
        """Handle symptom extraction with enhanced processing."""
        start_time = time.time()
        
        try:
            if 'extract' in task.lower():
                result = await self._extract_symptoms(context)
            elif 'batch' in task.lower():
                result = await self._batch_extract_symptoms(context)
            elif 'analyze' in task.lower():
                result = await self._analyze_symptoms(context)
            elif 'enhance' in task.lower():
                result = await self._enhance_symptom_description(context)
            else:
                result = await self._general_symptom_processing(task, context)
            
            # Track performance
            processing_time = time.time() - start_time
            self.extraction_metrics['processing_times'].append(processing_time)
            self.extraction_metrics['total_extractions'] += 1
            
            if result.get('success', False):
                self.extraction_metrics['successful_extractions'] += 1
            
            return await self.post_process(result, context)
            
        except Exception as e:
            await self.logger.error(f"Error in symptom extraction: {e}")
            return {
                'success': False,
                'error': str(e),
                'symptoms': [],
                'processing_time': time.time() - start_time
            }
    
    async def _extract_symptoms(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract symptoms from text with enhanced processing."""
        text = context.get('text', '')
        if not text:
            return {'success': False, 'error': 'No text provided', 'symptoms': []}
        
        # Check cache first
        cache_key = f"extract_symptoms:{hash(text)}"
        cached_result = await self.get_cached_result(cache_key)
        if cached_result is not None:
            self.extraction_metrics['cache_hits'] += 1
            return cached_result
        
        symptoms = []
        confidence_scores = []
        
        # 1. Pattern-based extraction
        pattern_symptoms = await self._extract_with_patterns(text)
        symptoms.extend(pattern_symptoms)
        
        # 2. Enhanced extraction with embedding service
        if 'embedding_service' in self.services:
            semantic_symptoms = await self._extract_with_embeddings(text)
            symptoms.extend(semantic_symptoms)
        
        # 3. Search-enhanced extraction
        if 'search_engine_service' in self.services:
            search_symptoms = await self._extract_with_search(text)
            symptoms.extend(search_symptoms)
        
        # 4. LLM-based extraction if chat service available
        if 'chat_service' in self.services:
            llm_symptoms = await self._extract_with_llm(text, context)
            symptoms.extend(llm_symptoms)
        
        # Deduplicate and score symptoms
        unique_symptoms = await self._deduplicate_symptoms(symptoms)
        scored_symptoms = await self._score_symptoms(unique_symptoms, text)
        
        # Filter by confidence threshold
        filtered_symptoms = [
            symptom for symptom in scored_symptoms 
            if symptom.get('confidence', 0) >= self.confidence_threshold
        ]
        
        result = {
            'success': True,
            'symptoms': filtered_symptoms,
            'total_found': len(symptoms),
            'unique_count': len(unique_symptoms),
            'high_confidence_count': len(filtered_symptoms),
            'extraction_methods': {
                'patterns': len(pattern_symptoms),
                'embeddings': len(semantic_symptoms) if 'embedding_service' in self.services else 0,
                'search': len(search_symptoms) if 'search_engine_service' in self.services else 0,
                'llm': len(llm_symptoms) if 'chat_service' in self.services else 0
            }
        }
        
        # Cache result
        await self.cache_result(cache_key, result, ttl=self.cache_ttl)
        
        return result
    
    async def _extract_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract symptoms using predefined patterns."""
        import re
        
        symptoms = []
        for pattern in self.symptom_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                symptoms.append({
                    'text': match.group(),
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'confidence': 0.8,  # High confidence for pattern matches
                    'method': 'pattern',
                    'pattern': pattern
                })
        
        return symptoms
    
    async def _extract_with_embeddings(self, text: str) -> List[Dict[str, Any]]:
        """Extract symptoms using embedding-based similarity."""
        try:
            embedding_service = self.services['embedding_service']
            
            # Create embeddings for the text
            text_embeddings = await embedding_service.create_embeddings([text])
            if not text_embeddings:
                return []
            
            # Create embeddings for known symptom patterns
            symptom_texts = [
                'engine knocking', 'brake squealing', 'transmission slipping',
                'steering vibration', 'suspension noise', 'electrical problems',
                'exhaust smoking', 'tire wear', 'oil leak', 'ac not working'
            ]
            
            symptom_embeddings = await embedding_service.create_embeddings(symptom_texts)
            if not symptom_embeddings:
                return []
            
            # Find similar symptoms using cosine similarity
            similarities = await embedding_service.compute_similarity_matrix(
                text_embeddings, symptom_embeddings
            )
            
            symptoms = []
            if similarities and len(similarities) > 0:
                for i, similarity in enumerate(similarities[0]):
                    if similarity > 0.6:  # Similarity threshold
                        symptoms.append({
                            'text': symptom_texts[i],
                            'confidence': float(similarity),
                            'method': 'embedding',
                            'similarity_score': float(similarity)
                        })
            
            return symptoms
            
        except Exception as e:
            await self.logger.error(f"Embedding-based extraction failed: {e}")
            return []
    
    async def _extract_with_search(self, text: str) -> List[Dict[str, Any]]:
        """Extract symptoms using search engine for automotive context."""
        try:
            search_service = self.services['search_engine_service']
            
            # Extract automotive keywords from text
            automotive_context = []
            for keyword in self.automotive_keywords:
                if keyword.lower() in text.lower():
                    automotive_context.append(keyword)
            
            if not automotive_context:
                return []
            
            # Search for related automotive issues
            search_query = f"automotive symptoms {' '.join(automotive_context[:3])}"
            search_results = await search_service.search_web_chunks(
                query=search_query,
                limit=5
            )
            
            symptoms = []
            if search_results:
                for result in search_results:
                    content = result.get('content', '')
                    if content:
                        # Extract potential symptoms from search results
                        pattern_symptoms = await self._extract_with_patterns(content)
                        for symptom in pattern_symptoms:
                            symptom['method'] = 'search'
                            symptom['confidence'] *= 0.7  # Lower confidence for search-derived
                            symptom['source'] = result.get('url', 'search')
                            symptoms.append(symptom)
            
            return symptoms
            
        except Exception as e:
            await self.logger.error(f"Search-based extraction failed: {e}")
            return []
    
    async def _extract_with_llm(self, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract symptoms using LLM through chat service."""
        try:
            chat_service = self.services['chat_service']
            
            # Enhance context for LLM
            enhanced_context = context.copy()
            enhanced_context.update({
                'task': 'symptom_extraction',
                'automotive_context': True,
                'extract_structured_data': True
            })
            
            # Create extraction prompt
            extraction_prompt = f"""
            Please extract automotive symptoms from the following text and return them as a JSON list.
            Each symptom should include:
            - text: the symptom description
            - category: the automotive system involved (engine, brake, transmission, etc.)
            - severity: estimated severity (low, medium, high)
            - confidence: your confidence in this being a real symptom (0.0-1.0)
            
            Text to analyze: {text}
            
            Return only the JSON array of symptoms.
            """
            
            response = await chat_service.process_message(
                message=extraction_prompt,
                context=enhanced_context
            )
            
            # Parse LLM response
            symptoms = []
            if response and 'content' in response:
                try:
                    import json
                    parsed_symptoms = json.loads(response['content'])
                    if isinstance(parsed_symptoms, list):
                        for symptom in parsed_symptoms:
                            if isinstance(symptom, dict) and 'text' in symptom:
                                symptom['method'] = 'llm'
                                symptom['confidence'] = symptom.get('confidence', 0.7)
                                symptoms.append(symptom)
                except json.JSONDecodeError:
                    await self.logger.warning("Failed to parse LLM response as JSON")
            
            return symptoms
            
        except Exception as e:
            await self.logger.error(f"LLM-based extraction failed: {e}")
            return []
    
    async def _deduplicate_symptoms(self, symptoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate symptoms and merge similar ones."""
        if not symptoms:
            return []
        
        # Simple deduplication by text similarity
        unique_symptoms = []
        seen_texts = set()
        
        for symptom in symptoms:
            text = symptom.get('text', '').lower().strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_symptoms.append(symptom)
            elif text in seen_texts:
                # Find existing symptom and update confidence if higher
                for existing in unique_symptoms:
                    if existing.get('text', '').lower().strip() == text:
                        existing_conf = existing.get('confidence', 0)
                        new_conf = symptom.get('confidence', 0)
                        if new_conf > existing_conf:
                            existing['confidence'] = new_conf
                        break
        
        return unique_symptoms
    
    async def _score_symptoms(self, symptoms: List[Dict[str, Any]], original_text: str) -> List[Dict[str, Any]]:
        """Score symptoms based on various factors."""
        for symptom in symptoms:
            base_confidence = symptom.get('confidence', 0.5)
            
            # Boost confidence for automotive-specific terms
            text = symptom.get('text', '').lower()
            automotive_boost = 0
            for keyword in self.automotive_keywords:
                if keyword.lower() in text:
                    automotive_boost += 0.1
            
            # Boost confidence if symptom appears in original text
            text_presence_boost = 0.2 if text in original_text.lower() else 0
            
            # Calculate final confidence
            final_confidence = min(1.0, base_confidence + automotive_boost + text_presence_boost)
            symptom['confidence'] = final_confidence
            
            # Add scoring metadata
            symptom['scoring'] = {
                'base_confidence': base_confidence,
                'automotive_boost': automotive_boost,
                'text_presence_boost': text_presence_boost,
                'final_confidence': final_confidence
            }
        
        # Sort by confidence
        symptoms.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return symptoms
    
    async def _batch_extract_symptoms(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract symptoms from multiple texts in batch."""
        texts = context.get('texts', [])
        if not texts:
            return {'success': False, 'error': 'No texts provided', 'results': []}
        
        # Limit batch size
        batch_size = min(len(texts), self.max_batch_size)
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        all_results = []
        
        for batch in batches:
            batch_tasks = []
            for i, text in enumerate(batch):
                batch_context = context.copy()
                batch_context['text'] = text
                batch_context['batch_index'] = i
                batch_tasks.append(self._extract_symptoms(batch_context))
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    all_results.append({
                        'success': False,
                        'error': str(result),
                        'text_index': i,
                        'symptoms': []
                    })
                else:
                    result['text_index'] = i
                    all_results.append(result)
        
        return {
            'success': True,
            'batch_size': len(texts),
            'processed_count': len(all_results),
            'results': all_results
        }
    
    async def _analyze_symptoms(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze extracted symptoms for patterns and insights."""
        symptoms = context.get('symptoms', [])
        if not symptoms:
            return {'success': False, 'error': 'No symptoms provided', 'analysis': {}}
        
        analysis = {
            'total_symptoms': len(symptoms),
            'confidence_distribution': {},
            'method_distribution': {},
            'category_distribution': {},
            'severity_distribution': {},
            'top_symptoms': symptoms[:5],  # Top 5 by confidence
            'automotive_systems': []
        }
        
        # Analyze confidence distribution
        confidence_ranges = {'high': 0, 'medium': 0, 'low': 0}
        for symptom in symptoms:
            confidence = symptom.get('confidence', 0)
            if confidence >= 0.8:
                confidence_ranges['high'] += 1
            elif confidence >= 0.5:
                confidence_ranges['medium'] += 1
            else:
                confidence_ranges['low'] += 1
        analysis['confidence_distribution'] = confidence_ranges
        
        # Analyze extraction methods
        methods = {}
        for symptom in symptoms:
            method = symptom.get('method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        analysis['method_distribution'] = methods
        
        # Analyze categories and severity if available
        categories = {}
        severities = {}
        systems = set()
        
        for symptom in symptoms:
            category = symptom.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
            severity = symptom.get('severity', 'unknown')
            severities[severity] = severities.get(severity, 0) + 1
            
            # Extract automotive systems
            text = symptom.get('text', '').lower()
            for keyword in ['engine', 'brake', 'transmission', 'steering', 'suspension', 'electrical']:
                if keyword in text:
                    systems.add(keyword)
        
        analysis['category_distribution'] = categories
        analysis['severity_distribution'] = severities
        analysis['automotive_systems'] = list(systems)
        
        return {
            'success': True,
            'analysis': analysis
        }
    
    async def _enhance_symptom_description(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance symptom descriptions with additional context."""
        symptoms = context.get('symptoms', [])
        if not symptoms:
            return {'success': False, 'error': 'No symptoms provided', 'enhanced_symptoms': []}
        
        enhanced_symptoms = []
        
        for symptom in symptoms:
            enhanced = symptom.copy()
            
            # Add automotive context if search service available
            if 'search_engine_service' in self.services:
                try:
                    symptom_text = symptom.get('text', '')
                    search_results = await self.services['search_engine_service'].search_web_chunks(
                        query=f"automotive {symptom_text} causes solutions",
                        limit=2
                    )
                    
                    if search_results:
                        enhanced['related_info'] = [
                            {
                                'source': result.get('url', 'unknown'),
                                'content_snippet': result.get('content', '')[:200] + '...'
                            }
                            for result in search_results[:2]
                        ]
                except Exception as e:
                    await self.logger.warning(f"Failed to enhance symptom '{symptom_text}': {e}")
            
            # Add automotive system mapping
            enhanced['automotive_system'] = self._map_to_automotive_system(symptom.get('text', ''))
            
            # Add potential causes (simplified logic)
            enhanced['potential_causes'] = self._suggest_potential_causes(symptom.get('text', ''))
            
            enhanced_symptoms.append(enhanced)
        
        return {
            'success': True,
            'enhanced_symptoms': enhanced_symptoms,
            'enhancement_count': len(enhanced_symptoms)
        }
    
    def _map_to_automotive_system(self, symptom_text: str) -> str:
        """Map symptom to automotive system."""
        text = symptom_text.lower()
        
        system_keywords = {
            'engine': ['engine', 'motor', 'combustion', 'cylinder', 'piston', 'spark', 'fuel', 'oil'],
            'transmission': ['transmission', 'gearbox', 'clutch', 'shift', 'gear'],
            'brake': ['brake', 'braking', 'stop', 'pad', 'rotor', 'disc'],
            'steering': ['steering', 'wheel', 'turn', 'direction'],
            'suspension': ['suspension', 'shock', 'strut', 'spring', 'bounce'],
            'electrical': ['battery', 'alternator', 'electrical', 'wire', 'fuse', 'light'],
            'exhaust': ['exhaust', 'muffler', 'pipe', 'emission'],
            'cooling': ['radiator', 'coolant', 'temperature', 'overheat'],
            'fuel': ['fuel', 'gas', 'diesel', 'pump', 'tank', 'carburetor', 'injector']
        }
        
        for system, keywords in system_keywords.items():
            if any(keyword in text for keyword in keywords):
                return system
        
        return 'unknown'
    
    def _suggest_potential_causes(self, symptom_text: str) -> List[str]:
        """Suggest potential causes for a symptom."""
        text = symptom_text.lower()
        
        cause_mapping = {
            'knocking': ['low octane fuel', 'carbon buildup', 'timing issues'],
            'squealing': ['worn brake pads', 'worn belt', 'bearing issues'],
            'overheating': ['coolant leak', 'thermostat failure', 'radiator blockage'],
            'vibration': ['wheel balance', 'alignment issues', 'worn suspension'],
            'slipping': ['low fluid', 'worn clutch', 'transmission wear'],
            'hard starting': ['battery weak', 'fuel pump', 'starter motor'],
            'rough idle': ['dirty air filter', 'spark plugs', 'fuel system']
        }
        
        for symptom_key, causes in cause_mapping.items():
            if symptom_key in text:
                return causes
        
        return ['requires professional diagnosis']
    
    async def _general_symptom_processing(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general symptom processing tasks."""
        # Default to symptom extraction
        return await self._extract_symptoms(context)
    
    async def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction performance statistics."""
        processing_times = self.extraction_metrics['processing_times']
        
        return {
            'total_extractions': self.extraction_metrics['total_extractions'],
            'successful_extractions': self.extraction_metrics['successful_extractions'],
            'success_rate': (
                self.extraction_metrics['successful_extractions'] / 
                self.extraction_metrics['total_extractions']
                if self.extraction_metrics['total_extractions'] > 0 else 0
            ),
            'cache_hits': self.extraction_metrics['cache_hits'],
            'cache_hit_rate': (
                self.extraction_metrics['cache_hits'] / 
                self.extraction_metrics['total_extractions']
                if self.extraction_metrics['total_extractions'] > 0 else 0
            ),
            'average_processing_time': (
                sum(processing_times) / len(processing_times)
                if processing_times else 0
            ),
            'total_patterns': len(self.symptom_patterns),
            'total_keywords': len(self.automotive_keywords)
        }


class DiagnosisTreeAgent:
    """
    Enhanced diagnosis tree agent for managing automotive diagnosis trees.
    Integrated with improved services for better performance and accuracy.
    """
    """
    Unified LangChain agent for managing a diagnosis tree with optional LLM-based expansion.
    - Maintains a tree of issues, each with a name, likelihood, and associated data.
    - Supports batch updates: incorporate new issues and update existing ones, inferring parent-child relationships.
    - Prunes low-likelihood issues and sorts children by likelihood.
    - Can expand any node (or the entire tree) using an LLMChain based on context and symptom text.
    - Thread-safe operations using an asyncio lock for concurrent usage.
    """
    def __init__(self, llm: BaseLanguageModel, prompt: PromptTemplate,
                 root_issue_name: str = "root", root_likelihood: float = 1.0,
                 root: Optional['DiagnosisTreeAgent.TreeNode'] = None,
                 logger: Optional[Logger] = None):
        self.lock = asyncio.Lock()
        if root is not None:
            self.root = root
            # Build node_map from the provided root's subtree
            self.node_map: Dict[str, DiagnosisTreeAgent.TreeNode] = {n.issue_name: n for n in root.traverse()}
        else:
            # Create the root node of the tree
            self.root = self._create_root(root_issue_name, root_likelihood)
            self.node_map: Dict[str, DiagnosisTreeAgent.TreeNode] = {self.root.issue_name: self.root}
        # LLM chain for expansions
        self.llm = llm
        self.prompt = prompt
        self.output_parser = JsonOutputParser()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.logger = logger or Logger()

    class TreeNode:
        """
        Internal class representing a node in the diagnosis tree.
        """
        def __init__(self, issue_name: str, likelihood: float, data: Any = None,
                     parent: Optional['DiagnosisTreeAgent.TreeNode'] = None):
            self.issue_name = issue_name
            self.likelihood = likelihood
            self.data = data
            self.children: List['DiagnosisTreeAgent.TreeNode'] = []
            self.parent: Optional['DiagnosisTreeAgent.TreeNode'] = parent

        def add_child(self, child: 'DiagnosisTreeAgent.TreeNode'):
            """
            Add a child node to this node.
            """
            child.parent = self
            self.children.append(child)

        def remove_child(self, child: 'DiagnosisTreeAgent.TreeNode'):
            """
            Remove a child node from this node.
            """
            self.children.remove(child)
            child.parent = None

        def find(self, issue_name: str) -> Optional['DiagnosisTreeAgent.TreeNode']:
            """
            Find a node by issue name in the subtree rooted at this node.
            """
            if self.issue_name == issue_name:
                return self
            for child in self.children:
                result = child.find(issue_name)
                if result:
                    return result
            return None

        def traverse(self) -> List['DiagnosisTreeAgent.TreeNode']:
            """
            Traverse the tree (pre-order) and return all nodes in a list.
            """
            nodes = [self]
            for child in self.children:
                nodes.extend(child.traverse())
            return nodes

        def update_data(self, data: Any):
            """
            Update the data associated with this node.
            """
            self.data = data

        def prune(self, threshold: float = 0.3):
            """
            Prune children whose likelihood is below the threshold (e.g., 0.3 for 30%).
            """
            pruned_children = []
            for child in self.children:
                if child.likelihood < threshold:
                    pruned_children.append(child)
                else:
                    child.prune(threshold)
            for child in pruned_children:
                self.children.remove(child)

        def sort_children_by_likelihood(self, reverse: bool = True):
            """
            Sort children nodes by their likelihood (descending by default).
            """
            self.children.sort(key=lambda x: x.likelihood, reverse=reverse)

        def process(self):
            """
            Placeholder method; override with processing logic if needed.
            """
            pass

    def _create_root(self, issue_name: str, likelihood: float) -> 'DiagnosisTreeAgent.TreeNode':
        """
        Create a root node (subclass of TreeNode) with the given name and likelihood.
        """
        root = DiagnosisTreeAgent.TreeNode(issue_name, likelihood)
        return root

    async def get_tree(self) -> TreeNode:
        """
        Get the root of the diagnosis tree.
        """
        async with self.lock:
            return self.root

    async def find_issue(self, issue_name: str) -> Optional[TreeNode]:
        """
        Find and return the node with the given issue_name, or None if not found.
        """
        async with self.lock:
            return self.node_map.get(issue_name)

    async def update_tree_from_nodes(self, nodes: List[TreeNode], prune_threshold: float = 0.3):
        """
        Batch update the tree with a list of nodes:
        - If a node exists, update its data and likelihood.
        - Otherwise, attach the new node to the tree (using parent reference if available, or as child of root).
        - After insertion, prune low-likelihood nodes and sort children by likelihood.
        """
        async with self.lock:
            for node in nodes:
                existing = self.node_map.get(node.issue_name)
                if existing:
                    # Update existing node's likelihood and data
                    existing.likelihood = node.likelihood
                    existing.update_data(node.data)
                    # If a new parent is provided and is different, re-attach node
                    if node.parent:
                        parent_name = node.parent.issue_name
                        parent = self.node_map.get(parent_name)
                        if parent and parent != existing.parent:
                            # Remove from old parent, add to new parent
                            if existing.parent:
                                existing.parent.remove_child(existing)
                            parent.add_child(existing)
                            existing.parent = parent
                else:
                    # Node does not exist; determine parent
                    parent = None
                    if node.parent:
                        # Check if parent already in current tree
                        parent = self.node_map.get(node.parent.issue_name)
                        if not parent:
                            # Parent is also new in this batch? Find the actual node object for parent
                            parent = next((n for n in nodes if n.issue_name == node.parent.issue_name), None)
                    if parent:
                        parent.add_child(node)
                        node.parent = parent
                    else:
                        # No parent specified or not found; attach to root
                        self.root.add_child(node)
                        node.parent = self.root
                    # Add this new node to the map
                    self.node_map[node.issue_name] = node
            # Prune low-likelihood nodes from the tree
            self.root.prune(prune_threshold)
            # Sort children by likelihood at each node
            for n in self.root.traverse():
                n.sort_children_by_likelihood()
            # Rebuild the node map to ensure it's consistent with the pruned tree
            self.node_map = {n.issue_name: n for n in self.root.traverse()}
        await self.logger.info(f"Tree updated from nodes. Total nodes: {len(self.node_map)}")

    async def update_issue(self, issue_name: str, data: Any):
        """
        Update the data of a single issue in the tree.
        """
        async with self.lock:
            node = self.node_map.get(issue_name)
            if not node:
                await self.logger.error(f"Issue '{issue_name}' not found.")
                raise ValueError(f"Issue '{issue_name}' not found.")
            node.update_data(data)
        await self.logger.info(f"Issue '{issue_name}' updated.")

    async def prune_tree(self, threshold: float = 0.3):
        """
        Prune the tree by removing all nodes (and subtrees) below the likelihood threshold.
        """
        async with self.lock:
            self.root.prune(threshold)
            self.node_map = {n.issue_name: n for n in self.root.traverse()}
        await self.logger.info(f"Tree pruned with threshold {threshold}.")

    async def sort_children(self, issue_name: str, reverse: bool = True):
        """
        Sort the children of the given node by likelihood.
        """
        async with self.lock:
            node = self.node_map.get(issue_name)
            if node:
                node.sort_children_by_likelihood(reverse=reverse)

    async def reset(self):
        """
        Reset the entire tree to just the root node, clearing all other issues.
        """
        async with self.lock:
            root_name = self.root.issue_name
            root_likelihood = self.root.likelihood
            self.root = self._create_root(root_name, root_likelihood)
            self.node_map = {self.root.issue_name: self.root}
        await self.logger.info("Diagnosis tree reset to root.")

    async def expand_node_with_llm(self, node_name: str, context: str, symptom_text: str) -> List[TreeNode]:
        """
        Expand a single node using the LLM chain:
        - Generates potential child issues based on context and symptom text.
        - Updates existing children or adds new ones, updating likelihood and data.
        - Sorts new children by likelihood before returning them.
        """
        async with self.lock:
            parent = self.node_map.get(node_name)
            if not parent:
                await self.logger.error(f"Node '{node_name}' not found.")
                raise ValueError(f"Node '{node_name}' not found.")
            llm_input = {
                "parent_issue": node_name,
                "context": context,
                "symptom_text": symptom_text
            }
            # Invoke LLM chain and parse JSON output
            output = self.chain.invoke(llm_input)
            try:
                parsed_issues = self.output_parser.parse(output)
            except Exception as e:
                await self.logger.error(f"LLM output parsing failed: {e}")
                parsed_issues = []
            new_children = []
            for issue in parsed_issues:
                name = issue.get("issue_name")
                likelihood = issue.get("likelihood", 0)
                data = issue.get("data", None)
                node = self.node_map.get(name)
                if node:
                    # Update existing child's likelihood and data
                    node.likelihood = likelihood
                    node.update_data(data)
                    # If the parent has changed, re-attach node to the new parent
                    if node.parent != parent:
                        if node.parent:
                            node.parent.remove_child(node)
                        parent.add_child(node)
                        node.parent = parent
                else:
                    # Create a new child node and attach it to the parent
                    new_node = DiagnosisTreeAgent.TreeNode(name, likelihood, data)
                    parent.add_child(new_node)
                    new_node.parent = parent
                    self.node_map[name] = new_node
                    new_children.append(new_node)
            # Sort the parent's children by likelihood (descending)
            parent.sort_children_by_likelihood()
        await self.logger.info(f"Node '{node_name}' expanded with LLM. Children count: {len(parent.children)}")
        # Return the list of (new or updated) child nodes for the given parent
        return [self.node_map.get(issue.get("issue_name")) for issue in parsed_issues if issue.get("issue_name") in self.node_map]

    async def expand_all_with_llm(self, context: str, symptom_text: str):
        """
        Recursively expand all nodes in the tree using the LLM chain.
        Each node will generate new child issues, which are then added to the tree.
        """
        async with self.lock:
            async def expand_node(node: DiagnosisTreeAgent.TreeNode):
                llm_input = {
                    "parent_issue": node.issue_name,
                    "context": context,
                    "symptom_text": symptom_text
                }
                output = self.chain.invoke(llm_input)
                try:
                    parsed_issues = self.output_parser.parse(output)
                except Exception as e:
                    await self.logger.error(f"LLM output parsing failed at node '{node.issue_name}': {e}")
                    parsed_issues = []
                for issue in parsed_issues:
                    name = issue.get("issue_name")
                    likelihood = issue.get("likelihood", 0)
                    data = issue.get("data", None)
                    child = self.node_map.get(name)
                    if child:
                        # Update existing child's likelihood and data
                        child.likelihood = likelihood
                        child.update_data(data)
                        if child.parent != node:
                            if child.parent:
                                child.parent.remove_child(child)
                            node.add_child(child)
                            child.parent = node
                    else:
                        # Add new child node to current node
                        new_node = DiagnosisTreeAgent.TreeNode(name, likelihood, data)
                        node.add_child(new_node)
                        new_node.parent = node
                        self.node_map[name] = new_node
                # Sort the current node's children
                node.sort_children_by_likelihood()
                # Recursively expand each child
                for child in list(node.children):
                    await expand_node(child)
            # Start expansion from the root
            await expand_node(self.root)
        await self.logger.info("All nodes expanded with LLM.")
