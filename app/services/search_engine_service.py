from app.services.base_service import BaseService
from app.services.embedding_service import EmbeddingService
from app.services.parser_service import ParserService
from app.services.scraper_service import ScraperService
from app.services.car_vectorization_service import CarVectorizationService
from app.db.milvus_handler import MilvusHandler
from app.CRUD.car_crud import CarCRUD
from typing import List, Dict, Any, Optional
import asyncio
import re
from langchain_core.documents import Document
from collections import OrderedDict
from app.core.interfaces import IWebSocketManager
from app.utils.message_types import MessageSource

class LRUCache:
    def __init__(self, capacity=32):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class SearchEngineService(BaseService):
    """
    Service for searching car-related documents and knowledge sources using vector search.
    """
    def __init__(
        self,
        websocket_manager: IWebSocketManager = None,
        embedding_service: Optional[EmbeddingService] = None,
        parser_service: Optional[ParserService] = None,
        scraper_service: Optional[ScraperService] = None,
        milvus_handler: Optional[MilvusHandler] = None,
        car_crud: Optional[CarCRUD] = None,
        vectorization_service: Optional[CarVectorizationService] = None,
        embedding_cache: Optional[LRUCache] = None,
    ):
        super().__init__(websocket_manager=websocket_manager)
        self.embedding_service = embedding_service or EmbeddingService(websocket_manager=websocket_manager)
        self.parser_service = parser_service or ParserService(websocket_manager=websocket_manager)
        self.scraper_service = scraper_service or ScraperService(websocket_manager=websocket_manager)
        self.milvus_handler = milvus_handler or MilvusHandler()
        self.car_crud = car_crud or CarCRUD()
        self.vectorization_service = vectorization_service or CarVectorizationService(websocket_manager=websocket_manager)
        # Cache for embeddings only - manual path cache no longer needed
        self._embedding_cache = embedding_cache or LRUCache(capacity=64)  # Increased capacity for better performance

    async def get_owner_manual_text(self, car_id: str) -> Optional[str]:
        """
        Get the owner's manual text for the given car_id from vectorized chunks.
        This method retrieves a limited number of chunks for efficiency.
        NOTE: This method should generally not be used - use embed_and_vector_search instead.
        """
        # PERFORMANCE WARNING: This method retrieves ALL chunks and combines them
        # It should only be used for specific cases where the full manual is needed
        if hasattr(self, 'logger') and self.logger:
            await self.logger.warning(f"get_owner_manual_text called for {car_id} - this may impact performance")
        
        # Get only a reasonable number of chunks instead of all of them
        chunks = await self.car_crud.get_owner_manual_chunks(car_id, query=None, top_k=50)  # Reduced from 1000
        if chunks:
            # Combine chunks into text (sorted by chunk_index for coherence)
            sorted_chunks = sorted(chunks, key=lambda x: x.get('chunk_index', 0))
            manual_text = '\n'.join([chunk.get('chunk', '') for chunk in sorted_chunks])
            return manual_text if manual_text.strip() else None
            
        return None
        
    def _normalize_car_id(self, car_id: str) -> str:
        """
        Normalize car ID to handle different formats and common variations.
        For example: "echo-toyota-2001" -> "toyota-echo-2001"
        """
        if not car_id:
            return car_id
            
        # Try to parse parts from the car_id
        parts = car_id.lower().strip().split('-')
        if len(parts) >= 3:
            # Check for common pattern where make and model are reversed
            # Common car makes that we can detect
            common_makes = ['toyota', 'honda', 'ford', 'chevrolet', 'bmw', 'audi', 'mercedes', 'nissan', 
                           'mazda', 'subaru', 'hyundai', 'kia', 'lexus', 'acura', 'volkswagen', 'vw']
            
            # If the first part isn't a known make but the second is, swap them
            if parts[0] not in common_makes and parts[1] in common_makes:
                model, make = parts[0], parts[1]
                remaining = '-'.join(parts[2:])
                return f"{make}-{model}-{remaining}"
        
        # If we couldn't normalize it, return the original
        return car_id

    @staticmethod
    def score_normalizer(scores, reverse=False):
        """
        Normalize a list of scores to [0, 1].
        If reverse=True, lower is better (e.g., L2 distance), otherwise higher is better (e.g., cosine similarity).
        """
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0 for _ in scores]
        if reverse:
            # Lower is better: invert
            return [(max_score - s) / (max_score - min_score) for s in scores]
        else:
            return [(s - min_score) / (max_score - min_score) for s in scores]

    async def embed_and_vector_search(self, car_id: str, query: str, top_k: int = 12, chunk_size: int = 800) -> List[Dict[str, Any]]:
        """
        Direct vector search for car manuals using Milvus.
        This completely replaces the old slow method that did on-demand text processing.
        """
        try:
            import time
            start_time = time.time()
            
            # Use the vectorization service to search directly in Milvus
            results = await self.vectorization_service.search_car_manual(
                query=query,
                car_id=car_id,
                top_k=top_k
            )
            
            elapsed = time.time() - start_time
            if hasattr(self, 'logger') and self.logger:
                await self.logger.info(f"Vector search for {car_id} took {elapsed:.2f}s, found {len(results)} results")
            
            # If no results, try normalized car_id
            if not results:
                normalized_car_id = self._normalize_car_id(car_id)
                if normalized_car_id != car_id:
                    start_time = time.time()
                    results = await self.vectorization_service.search_car_manual(
                        query=query,
                        
                        car_id=normalized_car_id,
                        top_k=top_k
                    )
                    elapsed = time.time() - start_time
                    if hasattr(self, 'logger') and self.logger:
                        await self.logger.info(f"Normalized vector search for {normalized_car_id} took {elapsed:.2f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                await self.logger.error(f"Error in vector search for car {car_id}: {str(e)}")
            return []

    async def vector_search_ground_knowledge(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Perform a vector search on ground knowledge for the query using Milvus.
        Enhanced to retrieve more comprehensive results from the 38,936 document knowledge base.
        """
        query_embedding = await self.embedding_service.embed_text(query)
        # Increase search results to leverage the large knowledge base more effectively
        milvus_results = self.milvus_handler.search(query_embedding, top_k=min(top_k, 100))
        
        results = []
        scores = []
        seen_content = set()  # Avoid duplicates
        
        for hits in milvus_results:
            for hit in hits:
                content = hit.entity.get("content_chunk", "")
                # Skip very short or duplicate content
                if len(content.strip()) < 50 or content in seen_content:
                    continue
                    
                results.append({
                    "source": "ground_knowledge",
                    "chunk": content,
                    "score": float(hit.distance),
                    "metadata": hit.entity.get("metadata", {}),
                    "book_title": hit.entity.get("book_title", ""),
                    "page_number": hit.entity.get("page_number", None),
                    "id": hit.entity.get("id", ""),
                })
                scores.append(float(hit.distance))
                seen_content.add(content)
        
        # Normalize scores (L2: lower is better)
        norm_scores = self.score_normalizer(scores, reverse=True)
        for i, norm_score in enumerate(norm_scores):
            results[i]["score"] = norm_score
        
        # Sort by normalized score (higher is better)
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def is_valid_url(self, url):
        return isinstance(url, str) and re.match(r'^(http://|https://|file://|raw:)', url.strip())

    async def scrape_and_vector_search_links(self, car_id: str, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        1. Use summaries and links from car_guide_links.
        2. Vector search the summaries to select the top 2 most relevant links.
        3. Scrape the content of those selected links and return them.
        """
        car = await self.car_crud.get_car_by_id(car_id)
        # car_guide_links should be a list of dicts: {"link": ..., "summary": ...}
        car_guide_links = car.get('car_guide_links', []) if car else []
        # Filter for valid links with summaries
        summaries = []
        links = []
        for entry in car_guide_links:
            link = entry.get('link') if isinstance(entry, dict) else None
            summary = entry.get('summary') if isinstance(entry, dict) else None
            if link and summary and isinstance(summary, str) and len(summary.strip()) > 0 and self.is_valid_url(link):
                links.append(link)
                summaries.append(summary)
        if not summaries:
            return []
        # Vector search the summaries to select top N links
        chunk_embeddings = await self.embedding_service.embed_texts_batch(summaries)
        query_embedding = await self.embedding_service.embed_text(query)
        top_matches = await self.embedding_service.find_most_similar(query_embedding, chunk_embeddings, top_k=top_k)
        top_indices = [idx for (idx, _) in top_matches]
        selected_links = [links[idx] for idx in top_indices]
        selected_summaries = [summaries[idx] for idx in top_indices]
        # Scrape the content of the selected links only
        detailed_results = await self.scraper_service.perform_action(links=selected_links, concurrency=2)
        results = []
        for i, detail in enumerate(detailed_results):
            results.append({
                "source": "car_guide_link",
                "url": selected_links[i],
                "summary": selected_summaries[i],
                "score": top_matches[i][1],
                "content": detail.get('text') or detail.get('content') or ""
            })
        return results

    async def search(self, car_id: str, query: str, top_k: int = 40) -> List[Document]:
        """
        PERFORMANCE OPTIMIZED: Comprehensive search with reduced scope for faster processing.
        Returns a list of LangChain Document objects.
        """
        # Run all three searches in parallel with OPTIMIZED limits for performance
        manual_task = self.embed_and_vector_search(car_id, query, top_k=max(8, top_k//5))  # Reduced scope
        ground_task = self.vector_search_ground_knowledge(query, top_k=max(25, int(top_k * 0.6)))  # Reduced from 0.75 to 0.6
        links_task = self.scrape_and_vector_search_links(car_id, query, top_k=max(3, top_k//15))  # Reduced scope
        
        manual_results, ground_results, link_results = await asyncio.gather(manual_task, ground_task, links_task)
        
        # Tag each result with its source type for normalization
        all_results = []
        
        # Prioritize manual results slightly
        for r in (manual_results or []):
            r['similarity_type'] = 'cosine'
            r['source_priority'] = 1.1  # Slight boost for manual
            all_results.append(r)
            
        # Knowledge base results (main focus)
        for r in (ground_results or []):
            r['similarity_type'] = 'l2'
            r['source_priority'] = 1.0
            all_results.append(r)
            
        # Online results
        for r in (link_results or []):
            r['similarity_type'] = 'cosine'
            r['source_priority'] = 0.9  # Slightly lower priority
            all_results.append(r)
        
        # Unify all scores to cosine-like (higher is better) and apply source priority
        for r in all_results:
            if r['similarity_type'] == 'l2':
                # Invert L2 so higher is better
                r['score'] = 1.0 - r['score']
            # Apply source priority
            r['score'] *= r.get('source_priority', 1.0)
        
        # Normalize all scores together
        scores = [r['score'] for r in all_results]
        norm_scores = self.score_normalizer(scores, reverse=False)
        for i, norm_score in enumerate(norm_scores):
            all_results[i]['score'] = norm_score
        
        # Sort by normalized score (higher is better)
        all_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)
        
        # Convert to LangChain Document objects, ensuring diversity
        documents = []
        seen_chunks = set()
        for doc in all_results:
            chunk = doc.get("chunk", "")
            # Skip very similar content to ensure diversity - IMPROVED filtering
            chunk_key = chunk[:200] if len(chunk) > 200 else chunk  # Use first 200 chars as key
            if len(chunk) > 50 and chunk_key not in seen_chunks:
                documents.append(Document(page_content=chunk, metadata={k: v for k, v in doc.items() if k != "chunk"}))
                seen_chunks.add(chunk_key)
                if len(documents) >= top_k:
                    break
        
        return documents

    async def perform_action(self, *args, websocket=None, session_id=None, **kwargs) -> Any:
        """
        Implementation of the abstract method from BaseService.
        Expects 'car_id' and 'query' in kwargs.
        """
        car_id = kwargs.get("car_id")
        query = kwargs.get("query")
        top_k = kwargs.get("top_k", 40)  # PERFORMANCE: Reduced default from 80 to 40
        if websocket:
            await self.send_ws_stage(websocket, "Search started", MessageSource.CHAT_SERVICE, session_id=session_id, details={"car_id": car_id, "query": query})
        if not car_id or not query:
            if websocket:
                await self.send_ws_error(websocket, "car_id and query are required", MessageSource.CHAT_SERVICE, session_id=session_id)
            raise ValueError("car_id and query are required")
        try:
            result = await self.search(car_id, query, top_k=top_k)
            if websocket:
                await self.send_ws_result(websocket, "Search complete", MessageSource.CHAT_SERVICE, session_id=session_id, details={"num_results": len(result)})
            return result
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.exception(f"SearchEngineService.perform_action error: {e}")
            if websocket:
                await self.send_ws_error(websocket, f"SearchEngineService.perform_action error: {e}", MessageSource.CHAT_SERVICE, session_id=session_id, details={"error": str(e)})
            raise

    async def aclose(self):
        """
        Explicit async cleanup for resource management.
        Call this before shutting down the app or event loop.
        """
        if hasattr(self.scraper_service, 'cleanup'):
            await self.scraper_service.cleanup()
