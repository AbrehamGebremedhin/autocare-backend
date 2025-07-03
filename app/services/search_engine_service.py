from app.services.base_service import BaseService
from app.services.embedding_service import EmbeddingService
from app.services.parser_service import ParserService
from app.services.scraper_service import ScraperService
from app.db.milvus_handler import MilvusHandler
from app.CRUD.car_crud import CarCRUD
from app.db.bucket_operations import SupabaseBucketManager
from typing import List, Dict, Any, Optional
import os
import asyncio
import re
import aiofiles
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
        bucket_manager: Optional[SupabaseBucketManager] = None,
        manual_path_cache: Optional[LRUCache] = None,
        embedding_cache: Optional[LRUCache] = None,
    ):
        super().__init__(websocket_manager=websocket_manager)
        self.embedding_service = embedding_service or EmbeddingService(websocket_manager=websocket_manager)
        self.parser_service = parser_service or ParserService(websocket_manager=websocket_manager)
        self.scraper_service = scraper_service or ScraperService(websocket_manager=websocket_manager)
        self.milvus_handler = milvus_handler or MilvusHandler()
        self.car_crud = car_crud or CarCRUD()
        self.bucket_manager = bucket_manager or SupabaseBucketManager()
        # Caches
        self._manual_path_cache = manual_path_cache or LRUCache(capacity=32)
        self._embedding_cache = embedding_cache or LRUCache(capacity=32)

    async def download_owner_manual(self, car_id: str) -> Optional[str]:
        """
        Download the owner's manual for the given car_id from the Supabase bucket.
        Returns the local file path to the manual PDF, or None if not found.
        If the manual text is available in the DB, returns None (should use text, not file).
        """
        # Prefer DB text over file
        manual_text = await self.car_crud.get_owner_manual_text(car_id)
        if manual_text:
            return None  # Indicate to use DB text, not file
        # Check cache first
        cached_path = self._manual_path_cache.get(car_id)
        if cached_path and os.path.exists(cached_path):
            return cached_path
        car = await self.car_crud.get_car_by_id(car_id)
        owner_manual_url = car.get('owner_manual_url') if car else None
        if not owner_manual_url:
            return None
        # owner_manual_url is expected to be in the format 'bucket_name/path/to/file.pdf'
        try:
            bucket_name, file_path = owner_manual_url.split('/', 1)
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.exception(f"Error splitting owner_manual_url: {owner_manual_url}")
            return None
        local_path = f"car_data/{car_id}_manual.pdf"
        # Download the file from Supabase if not already present locally
        if not os.path.exists(local_path):
            try:
                file_bytes = await self.bucket_manager.download_file(bucket_name, file_path)
                if not file_bytes:
                    return None
                async with aiofiles.open(local_path, 'wb') as f:
                    await f.write(file_bytes)
            except Exception as e:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.exception(f"Error downloading or saving manual for car_id {car_id}: {e}")
                return None
        self._manual_path_cache.set(car_id, local_path)
        return local_path

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

    async def embed_and_vector_search(self, content_path: str, query: str, top_k: int = 12, chunk_size: int = 800) -> List[Dict[str, Any]]:
        """
        Embed the content (PDF path or DB text) and perform a vector search against the query.
        Returns top_k relevant results.
        chunk_size is tunable for retrieval quality and speed.
        """
        # Try to get manual text from DB first
        car_id = None
        if content_path and content_path.startswith("car_data/") and content_path.endswith("_manual.pdf"):
            car_id = content_path[len("car_data/"):-len("_manual.pdf")]
        manual_text = await self.car_crud.get_owner_manual_text(car_id) if car_id else None
        if manual_text:
            # Use DB text, chunk and embed
            chunks = await self.parser_service.chunk_text_optimized(manual_text, chunk_size=chunk_size)
            if not chunks:
                return []
            chunk_embeddings = await self.embedding_service.embed_texts_batch(chunks)
            query_embedding = await self.embedding_service.embed_text(query)
            top_matches = await self.embedding_service.find_most_similar(query_embedding, chunk_embeddings, top_k=top_k)
            scores = [score for _, score in top_matches]
            norm_scores = self.score_normalizer(scores, reverse=False)
            results = [
                {"source": "owner_manual", "chunk": chunks[idx], "score": norm_score}
                for (idx, _), norm_score in zip(top_matches, norm_scores)
            ]
            return results
        # Use cache key based on file path, query, and chunk_size
        cache_key = f"{content_path}:{query}:{chunk_size}"
        cached_result = self._embedding_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        # Precompute and store embeddings for static documents (owner manuals)
        static_embedding_key = f"static:{content_path}:{chunk_size}"
        static_data = self._embedding_cache.get(static_embedding_key)
        if static_data is not None:
            chunks, chunk_embeddings = static_data
        else:
            async with aiofiles.open(content_path, 'rb') as f:
                pdf_bytes = await f.read()
            chunks = await self.parser_service.parse_pdf_bytes_optimized(pdf_bytes, chunk_size=chunk_size)
            if not chunks:
                return []
            chunk_embeddings = await self.embedding_service.embed_texts_batch(chunks)
            self._embedding_cache.set(static_embedding_key, (chunks, chunk_embeddings))
        query_embedding = await self.embedding_service.embed_text(query)
        top_matches = await self.embedding_service.find_most_similar(query_embedding, chunk_embeddings, top_k=top_k)
        scores = [score for _, score in top_matches]
        norm_scores = self.score_normalizer(scores, reverse=False)
        results = [
            {"source": "owner_manual", "chunk": chunks[idx], "score": norm_score}
            for (idx, _), norm_score in zip(top_matches, norm_scores)
        ]
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
        manual_path = await self.download_owner_manual(car_id)
        manual_task = self.embed_and_vector_search(manual_path, query, top_k=max(15, top_k//4)) if manual_path else asyncio.create_task(asyncio.sleep(0, result=[]))
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
