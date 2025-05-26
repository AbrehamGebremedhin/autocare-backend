from app.services.base_service import BaseService
from googlesearch import search
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from app.core.config import get_settings
from app.services.query_builder_service import QueryBuilderService
from typing import List, Dict, Any, Optional
import asyncio
from app.utils.logger import Logger

class SearchEngineService(BaseService):
    """
    Service to perform searches using optimized queries for vector search, YouTube, and search engines.
    Provides unified access to web, YouTube, and vector search functionalities.
    """
    def __init__(self, 
                 query_builder: QueryBuilderService = None,
                 youtube_client=None,
                 embedding_service=None,
                 parser_service=None,
                 scraper_service=None,
                 bucket_manager=None,
                 fetch_car_data_service=None,
                 db_handler=None,
                 logger: Optional[Logger] = None):
        """
        Initialize the SearchEngineService with required dependencies and API clients.
        Allows dependency injection for easier testing and flexibility.
        Args:
            query_builder: QueryBuilderService instance.
            youtube_client: YouTube API client.
            embedding_service: EmbeddingService instance.
            parser_service: ParserService instance.
            scraper_service: ScraperService instance.
            bucket_manager: SupabaseBucketManager instance.
            fetch_car_data_service: FetchCarDataService instance.
            db_handler: SupabaseDBHandler instance.
            logger: Logger instance.
        """
        super().__init__()
        self.logger = logger or Logger("SearchEngineService")
        settings = get_settings()
        self.query_builder = query_builder or QueryBuilderService()
        self.youtube = youtube_client or build('youtube', 'v3', developerKey=settings.YOUTUBE_API_KEY)
        from app.services.embedding_service import EmbeddingService
        from app.services.parser_service import ParserService
        from app.services.scraper_service import ScraperService
        from app.db.bucket_operations import SupabaseBucketManager
        from app.services.fetch_car_data_service import FetchCarDataService
        from app.db.base import SupabaseDBHandler
        self.embedding_service = embedding_service or EmbeddingService()
        self.parser_service = parser_service or ParserService()
        self.scraper_service = scraper_service or ScraperService()
        self.bucket_manager = bucket_manager or SupabaseBucketManager()
        self.fetch_car_data_service = fetch_car_data_service or FetchCarDataService()
        self.db_handler = db_handler or SupabaseDBHandler()

    async def _get_ground_knowledge_chunks(self, limit=50):
        """
        Retrieve ground knowledge chunks from the database with optimized query.
        Args:
            limit (int): Maximum number of chunks to retrieve.
        Returns:
            List[dict]: List of ground knowledge chunks.
        Raises:
            Exception: If retrieval fails.
        """
        try:
            # Use caching for frequently accessed ground knowledge
            cache_key = f"ground_knowledge_{limit}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            db = self.db_handler._client
            ground_knowledge = db.table("Ground_Knowledge").select("*").limit(limit).execute()
            ground_docs = ground_knowledge.data if hasattr(ground_knowledge, 'data') else ground_knowledge["data"]
            
            result = [
                {
                    "id": doc.get("id"),
                    "source": "ground_knowledge",
                    "content": doc.get("content_chunk"),
                    "vector": doc.get("vector"),
                    "metadata": doc.get("metadata", {}),
                }
                for doc in ground_docs
            ]
              # Cache result for 10 minutes
            self._set_cache(cache_key, result, 600)
            return result
        except Exception as e:
            self.logger.error(f"_get_ground_knowledge_chunks error: {e}")
            raise

    async def _get_owner_manual_chunks(self, make, model, year, limit=50):
        """
        Retrieve owner manual chunks for a specific car with caching.
        Args:
            make (str): Car make.
            model (str): Car model.
            year (int): Car year.
            limit (int): Maximum number of chunks.
        Returns:
            List[dict]: List of owner manual chunks.
        Raises:
            Exception: If retrieval fails.
        """
        try:
            # Cache manual chunks to avoid re-processing
            cache_key = f"manual_{make}_{model}_{year}_{limit}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            pdf_name = f"{make}-{model}_{year}_EN_US.pdf"
            bucket_name = "manuals"
            pdf_bytes = await self.bucket_manager.download_file(bucket_name, pdf_name)
            owner_chunks_list = await self.parser_service.perform_action(pdf_bytes, source_type="pdf", chunk_size=1000)
            
            result = [
                {"id": f"owner_{i}", "source": "owner_manual", "content": chunk, "vector": None, "metadata": {}}
                for i, chunk in enumerate(owner_chunks_list[:limit])
            ]
            
            # Cache manual chunks for 30 minutes
            self._set_cache(cache_key, result, 1800)
            return result
            bucket_name = "manuals"
            pdf_bytes = await self.bucket_manager.download_file(bucket_name, pdf_name)
            owner_chunks_list = await self.parser_service.perform_action(pdf_bytes, source_type="pdf", chunk_size=1000)
            return [
                {"id": f"owner_{i}", "source": "owner_manual", "content": chunk, "vector": None, "metadata": {}}
                for i, chunk in enumerate(owner_chunks_list[:limit])
            ]
        except Exception as e:
            await self.logger.error(f"_get_owner_manual_chunks error: {e}")
            raise

    async def _get_web_chunks(self, make, model, year, limit_links=10, limit_chunks=100):
        """
        Retrieve web chunks for a specific car.
        Args:
            make (str): Car make.
            model (str): Car model.
            year (int): Car year.
            limit_links (int): Max number of links to scrape.
            limit_chunks (int): Max number of chunks.
        Returns:
            List[dict]: List of web chunks.
        Raises:
            Exception: If retrieval fails.
        """
        try:
            links = await self.fetch_car_data_service.perform_action(make, model, year)
            scraped = await self.scraper_service.perform_action(links, limit=limit_links)
            web_chunks = []
            for i, page in enumerate(scraped):
                if 'text' in page and page['text']:
                    chunks = await self.parser_service.perform_action(page['text'], source_type="string", chunk_size=1000)
                    for j, chunk in enumerate(chunks):
                        if len(web_chunks) < limit_chunks:
                            web_chunks.append({
                                "id": f"web_{i}_{j}", "source": "web", "content": chunk, "vector": None, "metadata": {"url": page.get("url")}
                            })
                        else:
                            break
                if len(web_chunks) >= limit_chunks:
                    break
            return web_chunks
        except Exception as e:
            await self.logger.error(f"_get_web_chunks error: {e}")
            raise

    async def _get_web_chunks_optimized(self, make, model, year, limit_links=10, limit_chunks=100):
        """
        Retrieve web chunks for a specific car with parallel processing and caching.
        """
        try:
            # Cache web chunks to avoid re-scraping
            cache_key = f"web_{make}_{model}_{year}_{limit_links}_{limit_chunks}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            # Parallel fetching and scraping
            links_task = asyncio.create_task(self.fetch_car_data_service.perform_action(make, model, year))
            links = await links_task
            
            # Use optimized scraping with reduced concurrency for stability
            scraped = await self.scraper_service.perform_action(links, limit=limit_links, concurrency=3)
            
            # Parallel processing of text chunks
            chunk_tasks = []
            for i, page in enumerate(scraped[:limit_links]):
                if 'text' in page and page['text']:
                    task = asyncio.create_task(
                        self.parser_service.perform_action(page['text'], source_type="string", chunk_size=1000)
                    )
                    chunk_tasks.append((i, page, task))
            
            web_chunks = []
            for i, page, task in chunk_tasks:
                try:
                    chunks = await task
                    for j, chunk in enumerate(chunks):
                        if len(web_chunks) < limit_chunks:
                            web_chunks.append({
                                "id": f"web_{i}_{j}", 
                                "source": "web", 
                                "content": chunk, 
                                "vector": None, 
                                "metadata": {"url": page.get("url"), "title": page.get("title", "")}
                            })
                        else:
                            break
                except Exception as e:
                    await self.logger.warning(f"Failed to process page {page.get('url', 'unknown')}: {e}")
                
                if len(web_chunks) >= limit_chunks:
                    break
            
            # Cache web chunks for 15 minutes
            self._set_cache(cache_key, web_chunks, 900)
            return web_chunks
        except Exception as e:
            await self.logger.error(f"_get_web_chunks_optimized error: {e}")
            raise

    @staticmethod
    def _cosine_sim(a, b):
        """
        Compute cosine similarity between two vectors.
        Args:
            a (list): First vector.
            b (list): Second vector.
        Returns:
            float: Cosine similarity score.
        """
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @staticmethod
    def _cosine_sim_batch(query_vec, vectors):
        """
        Compute cosine similarity between query and multiple vectors efficiently.
        Args:
            query_vec (np.array): Query vector.
            vectors (np.array): Matrix of vectors.
        Returns:
            np.array: Similarity scores.
        """
        import numpy as np
        query_vec = np.array(query_vec)
        vectors = np.array(vectors)
        
        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(vectors_norm, query_norm)
        return similarities

    async def _embed_and_score(self, query, chunks):
        """
        Embed the query and chunks, then score them by similarity.
        Args:
            query (str): The search query.
            chunks (list): List of chunks to score.
        Returns:
            list: Chunks with similarity scores.
        Raises:
            Exception: If embedding or scoring fails.
        """
        try:
            query_vec = await self.embedding_service.embed_text(query)
            to_embed = [c["content"] for c in chunks if c["vector"] is None]
            if to_embed:
                vectors = await self.embedding_service.embed_texts(to_embed)
                vi = 0
                for c in chunks:
                    if c["vector"] is None:
                        c["vector"] = vectors[vi]
                        vi += 1
            for c in chunks:
                c["score"] = self._cosine_sim(query_vec, c["vector"])
            return chunks
        except Exception as e:
            await self.logger.error(f"_embed_and_score error: {e}")
            raise

    async def _embed_and_score_optimized(self, query, chunks):
        """
        Optimized embedding and scoring with batch processing.
        """
        try:
            if not chunks:
                return []

            # Get query embedding
            query_vec = await self.embedding_service.embed_text(query)
            
            # Separate chunks with and without embeddings
            chunks_to_embed = []
            chunks_with_embeddings = []
            
            for chunk in chunks:
                if chunk["vector"] is None:
                    chunks_to_embed.append(chunk)
                else:
                    chunks_with_embeddings.append(chunk)
            
            # Batch embed missing vectors
            if chunks_to_embed:
                texts_to_embed = [c["content"] for c in chunks_to_embed]
                vectors = await self.embedding_service.embed_texts_batch(texts_to_embed)
                
                for i, chunk in enumerate(chunks_to_embed):
                    chunk["vector"] = vectors[i]
            
            # Combine all chunks
            all_chunks = chunks_with_embeddings + chunks_to_embed
            
            # Batch compute similarities
            if all_chunks:
                all_vectors = [c["vector"] for c in all_chunks]
                similarities = self._cosine_sim_batch(query_vec, all_vectors)
                
                for i, chunk in enumerate(all_chunks):
                    chunk["score"] = float(similarities[i])
            
            return all_chunks
        except Exception as e:
            await self.logger.error(f"_embed_and_score_optimized error: {e}")
            raise

    async def vector_search(self, query: str, query_type: str = None, make: str = None, model: str = None, year: int = None) -> list:
        """
        Perform a vector search across ground knowledge DB, owner manual PDF, and scraped web links.
        Args:
            query (str): The search query.
            query_type (str): 'generation' or 'validation'.
            make, model, year: Car info for owner manual and guides.
        Returns:
            List of ranked search results.
        Raises:
            Exception: If search fails.
        """
        try:
            if query_type == "generation":
                owner_chunks = await self._get_owner_manual_chunks(make, model, year) if (make and model and year) else []
                web_chunks = await self._get_web_chunks(make, model, year) if (make and model and year) else []
                all_chunks = owner_chunks + web_chunks
                all_chunks = await self._embed_and_score(query, all_chunks)
                all_chunks.sort(key=lambda x: x["score"], reverse=True)
                return all_chunks[:72]
            elif query_type == "validation":
                ground_chunks = await self._get_ground_knowledge_chunks()
                ground_chunks = await self._embed_and_score(query, ground_chunks)
                ground_chunks.sort(key=lambda x: x["score"], reverse=True)
                return ground_chunks
            else:
                ground_chunks = await self._get_ground_knowledge_chunks()
                owner_chunks = await self._get_owner_manual_chunks(make, model, year) if (make and model and year) else []
                web_chunks = await self._get_web_chunks(make, model, year) if (make and model and year) else []
                all_chunks = ground_chunks + owner_chunks + web_chunks
                all_chunks = await self._embed_and_score(query, all_chunks)
                all_chunks.sort(key=lambda x: x["score"], reverse=True)
                return all_chunks
        except Exception as e:
            await self.logger.error(f"vector_search error: {e}")
            raise

    async def vector_search_optimized(self, query: str, query_type: str = None, make: str = None, model: str = None, year: int = None) -> list:
        """
        Optimized vector search with parallel processing and improved ranking.
        """
        try:
            if query_type == "generation":
                # Parallel fetching of different data sources
                tasks = []
                if make and model and year:
                    tasks.append(asyncio.create_task(self._get_owner_manual_chunks(make, model, year)))
                    tasks.append(asyncio.create_task(self._get_web_chunks_optimized(make, model, year)))
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    owner_chunks = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else []
                    web_chunks = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
                else:
                    owner_chunks, web_chunks = [], []
                
                all_chunks = owner_chunks + web_chunks
                all_chunks = await self._embed_and_score_optimized(query, all_chunks)
                
                # Enhanced ranking with source weights
                for chunk in all_chunks:
                    if chunk["source"] == "owner_manual":
                        chunk["score"] *= 1.2  # Boost manual content
                    elif chunk["source"] == "web":
                        chunk["score"] *= 1.0  # Standard web content
                
                all_chunks.sort(key=lambda x: x["score"], reverse=True)
                return all_chunks[:72]
                
            elif query_type == "validation":
                ground_chunks = await self._get_ground_knowledge_chunks()
                ground_chunks = await self._embed_and_score_optimized(query, ground_chunks)
                ground_chunks.sort(key=lambda x: x["score"], reverse=True)
                return ground_chunks
                
            else:
                # Parallel fetching of all sources
                tasks = [
                    asyncio.create_task(self._get_ground_knowledge_chunks())
                ]
                
                if make and model and year:
                    tasks.extend([
                        asyncio.create_task(self._get_owner_manual_chunks(make, model, year)),
                        asyncio.create_task(self._get_web_chunks_optimized(make, model, year))
                    ])
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                ground_chunks = results[0] if not isinstance(results[0], Exception) else []
                owner_chunks = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
                web_chunks = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else []
                
                all_chunks = ground_chunks + owner_chunks + web_chunks
                all_chunks = await self._embed_and_score_optimized(query, all_chunks)
                
                # Apply source-based ranking
                for chunk in all_chunks:
                    if chunk["source"] == "ground_knowledge":
                        chunk["score"] *= 1.3  # Highest priority for ground knowledge
                    elif chunk["source"] == "owner_manual":
                        chunk["score"] *= 1.2
                    elif chunk["source"] == "web":
                        chunk["score"] *= 1.0
                
                all_chunks.sort(key=lambda x: x["score"], reverse=True)
                return all_chunks
                
        except Exception as e:
            await self.logger.error(f"vector_search_optimized error: {e}")
            raise

    async def web_search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a web search using the googlesearch-python package.
        Args:
            query (str): The search query.
            num_results (int): Number of results to return.
        Returns:
            List[Dict[str, Any]]: List of search result URLs.
        Raises:
            Exception: If search fails.
        """
        try:
            print(f"Performing web search for query: {query}")
            results = []
            loop = asyncio.get_running_loop()
            search_results = await loop.run_in_executor(None, lambda: list(search(query, num_results=num_results, advanced=True)))
            for result in search_results:
                results.append(result.url)
            return results
        except Exception as e:
            await self.logger.error(f"web_search error: {e}")
            return []
    
    async def youtube_search(self, query: str, max_results: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for videos using YouTube Data API with duration and quality filters.
        Args:
            query (str): The search query.
            max_results (int): Maximum number of results to return.
        Returns:
            List[Dict[str, Any]]: List of video information dictionaries.
        Raises:
            Exception: If search fails.
        """
        try:
            loop = asyncio.get_running_loop()
            def _search():
                search_response = self.youtube.search().list(
                    q=query,
                    part='id,snippet',
                    maxResults=max_results,
                    type='video',
                    videoDefinition='high',  # Only HD videos
                    videoEmbeddable='true',
                    videoLicense='youtube',
                    videoType='any'
                ).execute()
                video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
                if not video_ids:
                    return []
                video_response = self.youtube.videos().list(
                    id=','.join(video_ids),
                    part='contentDetails,statistics,snippet,status,player'
                ).execute()
                videos = []
                for item in video_response.get('items', []):
                    duration_str = item['contentDetails']['duration']
                    duration_seconds = self._parse_duration(duration_str)
                    videos.append({
                        'url': f'https://www.youtube.com/watch?v={item["id"]}',
                        'title': item['snippet']['title'],
                        'description': item['snippet'].get('description', ''),
                        'thumbnail': item['snippet'].get('thumbnails', {}).get('high', {}).get('url', ''),
                    })
                return videos
            return await loop.run_in_executor(None, _search)
        except HttpError as e:
            await self.logger.error(f"YouTube API error: {e}")
            return []
        except Exception as e:
            await self.logger.error(f"youtube_search error: {e}")
            return []

    def _parse_duration(self, duration_str: str) -> int:
        """
        Parse an ISO 8601 duration string (e.g., 'PT1H2M3S') to seconds.
        Args:
            duration_str (str): Duration string in ISO 8601 format.
        Returns:
            int: Duration in seconds.
        """
        import re
        pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
        match = pattern.match(duration_str)
        if not match:
            return 0
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else 0
        return hours * 3600 + minutes * 60 + seconds

    async def perform_action(self, user_query: str, query_type: str = None):
        """
        Perform a search action based on the query type.
        Args:
            user_query (str): The user's search query.
            query_type (str, optional): The type of search ('search_engine', 'youtube', 'vector').
        Returns:
            List of search results from the selected search method.
        Raises:
            Exception: If search fails.
        """
        try:
            query = await self.query_builder.perform_action(user_query, query_type)
            query = query.get('query', user_query)
            if not query:
                await self.logger.warning("No optimized query generated.")
                return []
            print(f"Optimized query: {query}")
            if not query_type:
                return await self.web_search(query)
            query_type = query_type.lower()
            if query_type == "search_engine":
                return await self.web_search(query)
            elif query_type == "youtube":
                return await self.youtube_search(query)
            elif query_type == "vector":
                return await self.vector_search(query)
            else:
                return await self.web_search(user_query)
        except Exception as e:
            await self.logger.error(f"perform_action error: {e}")
            raise
