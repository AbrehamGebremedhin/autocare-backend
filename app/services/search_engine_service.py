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
from langchain_core.documents import Document

class SearchEngineService(BaseService):
    """
    Service for searching car-related documents and knowledge sources using vector search.
    """
    def __init__(self, websocket_manager=None):
        super().__init__(websocket_manager=websocket_manager)
        self.embedding_service = EmbeddingService(websocket_manager=websocket_manager)
        self.parser_service = ParserService(websocket_manager=websocket_manager)
        self.scraper_service = ScraperService(websocket_manager=websocket_manager)
        self.milvus_handler = MilvusHandler()
        self.car_crud = CarCRUD()
        self.bucket_manager = SupabaseBucketManager()

    async def download_owner_manual(self, car_id: str) -> Optional[str]:
        """
        Download the owner's manual for the given car_id from the Supabase bucket.
        Returns the local file path to the manual PDF, or None if not found.
        """
        car = await self.car_crud.get_car_by_id(car_id)
        owner_manual_url = car.get('owner_manual_url') if car else None
        if not owner_manual_url:
            return None
        # owner_manual_url is expected to be in the format 'bucket_name/path/to/file.pdf'
        try:
            bucket_name, file_path = owner_manual_url.split('/', 1)
        except Exception:
            return None
        local_path = f"car_data/{car_id}_manual.pdf"
        # Download the file from Supabase if not already present locally
        if not os.path.exists(local_path):
            file_bytes = await self.bucket_manager.download_file(bucket_name, file_path)
            if not file_bytes:
                return None
            with open(local_path, 'wb') as f:
                f.write(file_bytes)
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

    async def embed_and_vector_search(self, content_path: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Embed the content (PDF path) and perform a vector search against the query.
        Returns top_k relevant results.
        """
        # 1. Parse PDF into chunks (improved chunking)
        with open(content_path, 'rb') as f:
            pdf_bytes = f.read()
        chunks = await self.parser_service.parse_pdf_bytes_optimized(pdf_bytes, chunk_size=1000)
        if not chunks:
            return []
        # 2. Embed chunks and query
        chunk_embeddings = await self.embedding_service.embed_texts_batch(chunks)
        query_embedding = await self.embedding_service.embed_text(query)
        # 3. Vector search
        top_matches = await self.embedding_service.find_most_similar(query_embedding, chunk_embeddings, top_k=top_k)
        # 4. Normalize scores (cosine similarity: higher is better)
        scores = [score for _, score in top_matches]
        norm_scores = self.score_normalizer(scores, reverse=False)
        results = [
            {"source": "owner_manual", "chunk": chunks[idx], "score": norm_score}
            for (idx, _), norm_score in zip(top_matches, norm_scores)
        ]
        return results

    async def vector_search_ground_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform a vector search on ground knowledge for the query using Milvus.
        """
        query_embedding = await self.embedding_service.embed_text(query)
        milvus_results = self.milvus_handler.search(query_embedding, top_k=top_k)
        results = []
        scores = []
        for hits in milvus_results:
            for hit in hits:
                results.append({
                    "source": "ground_knowledge",
                    "chunk": hit.entity.get("content_chunk", ""),
                    "score": float(hit.distance),
                    "metadata": hit.entity.get("metadata", {}),
                    "book_title": hit.entity.get("book_title", ""),
                    "page_number": hit.entity.get("page_number", None),
                })
                scores.append(float(hit.distance))
        # Normalize scores (L2: lower is better)
        norm_scores = self.score_normalizer(scores, reverse=True)
        for i, norm_score in enumerate(norm_scores):
            results[i]["score"] = norm_score
        # Sort by normalized score (higher is better)
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def is_valid_url(self, url):
        return isinstance(url, str) and re.match(r'^(http://|https://|file://|raw:)', url.strip())

    async def scrape_and_vector_search_links(self, car_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Scrape relevant links from car_guide_links and perform a vector search on their content.
        """
        # Retrieve car object and get car_guide_links
        car = await self.car_crud.get_car_by_id(car_id)
        car_guide_links = car.get('car_guide_links', []) if car else []
        # Filter for valid URLs
        car_guide_links = [url for url in car_guide_links if self.is_valid_url(url)]
        if not car_guide_links:
            return []
        # Scrape the links using ScraperService
        scraped_results = await self.scraper_service.perform_action(links=car_guide_links, limit=5, concurrency=3)
        # Extract text content from scraped results
        scraped_texts = [r.get('text', '') for r in scraped_results if r.get('text')]
        if not scraped_texts:
            return []
        chunk_embeddings = await self.embedding_service.embed_texts_batch(scraped_texts)
        query_embedding = await self.embedding_service.embed_text(query)
        top_matches = await self.embedding_service.find_most_similar(query_embedding, chunk_embeddings, top_k=top_k)
        scores = [score for _, score in top_matches]
        norm_scores = self.score_normalizer(scores, reverse=False)
        results = [
            {
                "source": "car_guide_link",
                "chunk": scraped_texts[idx],
                "score": norm_score,
                "url": car_guide_links[idx] if idx < len(car_guide_links) else None
            }
            for (idx, _), norm_score in zip(top_matches, norm_scores)
        ]
        return results

    async def search(self, car_id: str, query: str, top_k: int = 5) -> List[Document]:
        """
        Perform a comprehensive search using owner's manual, ground knowledge, and car guide links.
        Returns a list of LangChain Document objects.
        """
        # Run all three searches in parallel
        manual_task = self.embed_and_vector_search(await self.download_owner_manual(car_id), query, top_k=top_k)
        ground_task = self.vector_search_ground_knowledge(query, top_k=top_k)
        links_task = self.scrape_and_vector_search_links(car_id, query, top_k=top_k)
        manual_results, ground_results, link_results = await asyncio.gather(manual_task, ground_task, links_task)
        # Aggregate and sort all results by normalized score
        all_results = (manual_results or []) + (ground_results or []) + (link_results or [])
        all_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)
        # Convert to LangChain Document objects
        documents = [
            Document(page_content=doc.pop("chunk"), metadata=doc)
            for doc in all_results[:top_k]
        ]
        return documents

    async def perform_action(self, *args, **kwargs) -> Any:
        """
        Implementation of the abstract method from BaseService.
        Expects 'car_id' and 'query' in kwargs.
        """
        car_id = kwargs.get("car_id")
        query = kwargs.get("query")
        top_k = kwargs.get("top_k", 72)
        if not car_id or not query:
            raise ValueError("car_id and query are required")
        return await self.search(car_id, query, top_k=top_k)

    async def aclose(self):
        """
        Explicit async cleanup for resource management.
        Call this before shutting down the app or event loop.
        """
        if hasattr(self.scraper_service, 'cleanup'):
            await self.scraper_service.cleanup()
        