from app.services.base_service import BaseService
from app.services.embedding_service import EmbeddingService
from app.services.parser_service import ParserService
from app.services.scraper_service import ScraperService
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
        embedding_cache: Optional[LRUCache] = None,
    ):
        super().__init__(websocket_manager=websocket_manager)
        self.embedding_service = embedding_service or EmbeddingService(websocket_manager=websocket_manager)
        self.parser_service = parser_service or ParserService(websocket_manager=websocket_manager)
        self.scraper_service = scraper_service or ScraperService(websocket_manager=websocket_manager)
        self.milvus_handler = milvus_handler or MilvusHandler()
        self.car_crud = car_crud or CarCRUD()
        # Cache for embeddings only - manual path cache no longer needed
        self._embedding_cache = embedding_cache or LRUCache(capacity=64)  # Increased capacity for better performance

    async def get_owner_manual_text(self, car_id: str) -> Optional[str]:
        """
        Get the owner's manual text for the given car_id from the database.
        This method eliminates redundant PDF downloading and parsing since text is already stored.
        Returns the manual text or None if not found.
        """
        # Try original car_id first
        manual_text = await self.car_crud.get_owner_manual_text(car_id)
        if manual_text:
            return manual_text
            
        # If not found, try with possible alternative formats
        normalized_car_id = self._normalize_car_id(car_id)
        if normalized_car_id != car_id:
            if hasattr(self, 'logger') and self.logger:
                await self.logger.info(f"Trying normalized car_id: {normalized_car_id}")
            manual_text = await self.car_crud.get_owner_manual_text(normalized_car_id)
            if manual_text:
                if hasattr(self, 'logger') and self.logger:
                    await self.logger.info(f"Found manual text using normalized car_id: {normalized_car_id}")
                return manual_text
        
        # If still no text, check for predefined manuals
        # Example: check if we have a generic manual for the make/model
        try:
            parts = car_id.lower().split('-')
            if len(parts) >= 3:
                make, model, year = parts[0], parts[1], parts[2]
                # Try general make-model without year
                general_id = f"{make}-{model}"
                
                if hasattr(self, 'logger') and self.logger:
                    await self.logger.info(f"Trying generic manual for: {general_id}")
                
                general_manual = await self.car_crud.get_owner_manual_text(general_id)
                if general_manual:
                    if hasattr(self, 'logger') and self.logger:
                        await self.logger.info(f"Found generic manual for: {general_id}")
                    return general_manual
                    
                # Try with different year variations for same model
                # This would help if we have a manual for a different year of the same model
                for offset in [-1, 1, -2, 2]:
                    alt_year = str(int(year) + offset)
                    alt_id = f"{make}-{model}-{alt_year}"
                    
                    if hasattr(self, 'logger') and self.logger:
                        await self.logger.info(f"Trying alternative year: {alt_id}")
                    
                    alt_manual = await self.car_crud.get_owner_manual_text(alt_id)
                    if alt_manual:
                        if hasattr(self, 'logger') and self.logger:
                            await self.logger.info(f"Found manual for alternative year: {alt_id}")
                        return alt_manual
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                await self.logger.error(f"Error trying alternative car IDs: {str(e)}")
        
        # If no text found after all attempts, log warning and return None
        if hasattr(self, 'logger') and self.logger:
            await self.logger.warning(f"No manual text found in database for car_id: {car_id} or any alternatives")
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
        Optimized embedding and vector search using pre-stored text from database.
        Eliminates redundant PDF downloading and parsing.
        Returns top_k relevant results.
        """
        # Get manual text directly from database - no more PDF processing
        manual_text = await self.get_owner_manual_text(car_id)
        if not manual_text:
            return []
        
        # Use cache key based on car_id, query, and chunk_size for better cache efficiency
        cache_key = f"manual_search:{car_id}:{hash(query)}:{chunk_size}"
        cached_result = self._embedding_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Precompute and cache embeddings for static documents (owner manuals) by car_id
        static_embedding_key = f"manual_embeddings:{car_id}:{chunk_size}"
        static_data = self._embedding_cache.get(static_embedding_key)
        
        if static_data is not None:
            chunks, chunk_embeddings = static_data
        else:
            # Chunk the text and create embeddings
            chunks = await self.parser_service.chunk_text_optimized(manual_text, chunk_size=chunk_size)
            if not chunks:
                return []
            chunk_embeddings = await self.embedding_service.embed_texts_batch(chunks)
            # Cache the embeddings for future use
            self._embedding_cache.set(static_embedding_key, (chunks, chunk_embeddings))
        
        # Perform vector search
        query_embedding = await self.embedding_service.embed_text(query)
        top_matches = await self.embedding_service.find_most_similar(query_embedding, chunk_embeddings, top_k=top_k)
        
        scores = [score for _, score in top_matches]
        norm_scores = self.score_normalizer(scores, reverse=False)
        
        results = [
            {"source": "owner_manual", "chunk": chunks[idx], "score": norm_score}
            for (idx, _), norm_score in zip(top_matches, norm_scores)
        ]
        
        # Cache the final results
        self._embedding_cache.set(cache_key, results)
        return results

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

    async def search(self, car_id: str, query: str, top_k: int = 80) -> List[Document]:
        """
        Perform a comprehensive search using owner's manual, ground knowledge, and car guide links.
        Enhanced to leverage the 38,936 document knowledge base more effectively.
        Returns a list of LangChain Document objects.
        """
        # Run all three searches in parallel with increased limits for knowledge base
        manual_text = await self.get_owner_manual_text(car_id)
        manual_task = self.embed_and_vector_search(car_id, query, top_k=max(15, top_k//4)) if manual_text else asyncio.create_task(asyncio.sleep(0, result=[]))
        ground_task = self.vector_search_ground_knowledge(query, top_k=max(60, int(top_k * 0.75)))  # Focus on knowledge base
        links_task = self.scrape_and_vector_search_links(car_id, query, top_k=max(5, top_k//15))
        
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
            # Skip very similar content to ensure diversity
            if len(chunk) > 50 and chunk not in seen_chunks:
                documents.append(Document(page_content=chunk, metadata={k: v for k, v in doc.items() if k != "chunk"}))
                seen_chunks.add(chunk)
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
        top_k = kwargs.get("top_k", 80)  # Increased default to leverage knowledge base better
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
