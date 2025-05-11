from app.services.base_service import BaseService
from googlesearch import search
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from app.core.config import get_settings
from app.services.query_builder_service import QueryBuilderService
from typing import List, Dict, Any
import asyncio

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
                 db_handler=None):
        """
        Initialize the SearchEngineService with required dependencies and API clients.
        Allows dependency injection for easier testing and flexibility.
        """
        super().__init__()
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
        db = self.db_handler._client
        ground_knowledge = db.table("Ground_Knowledge").select("*").limit(limit).execute()
        ground_docs = ground_knowledge.data if hasattr(ground_knowledge, 'data') else ground_knowledge["data"]
        return [
            {
                "id": doc.get("id"),
                "source": "ground_knowledge",
                "content": doc.get("content_chunk"),
                "vector": doc.get("vector"),
                "metadata": doc.get("metadata", {}),
            }
            for doc in ground_docs
        ]

    async def _get_owner_manual_chunks(self, make, model, year, limit=50):
        pdf_name = f"{make}-{model}_{year}_EN_US.pdf"
        bucket_name = "manuals"
        pdf_bytes = await self.bucket_manager.download_file(bucket_name, pdf_name)
        owner_chunks_list = await self.parser_service.perform_action(pdf_bytes, source_type="pdf", chunk_size=1000)
        return [
            {"id": f"owner_{i}", "source": "owner_manual", "content": chunk, "vector": None, "metadata": {}}
            for i, chunk in enumerate(owner_chunks_list[:limit])
        ]

    async def _get_web_chunks(self, make, model, year, limit_links=10, limit_chunks=100):
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

    @staticmethod
    def _cosine_sim(a, b):
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def _embed_and_score(self, query, chunks):
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

    async def vector_search(self, query: str, query_type: str = None, make: str = None, model: str = None, year: int = None) -> list:
        """
        Perform a vector search across ground knowledge DB, owner manual PDF, and scraped web links.
        Args:
            query (str): The search query.
            query_type (str): 'generation' or 'validation'.
            make, model, year: Car info for owner manual and guides.
        Returns:
            List of ranked search results.
        """
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

    async def web_search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a web search using the googlesearch-python package.
        Args:
            query (str): The search query.
            num_results (int): Number of results to return.
        Returns:
            List[Dict[str, Any]]: List of search result dictionaries with 'title', 'link', and 'description'.
        """
        print(f"Performing web search for query: {query}")
        results = []
        try:
            loop = asyncio.get_running_loop()
            # googlesearch is not async, so run in executor
            search_results = await loop.run_in_executor(None, lambda: list(search(query, num_results=num_results, advanced=True)))
            for result in search_results:
                results.append({
                    'title': result.title,
                    'link': result.url,
                    'description': result.description
                })
        except Exception as e:
            print(f"Error during Google search: {e}")
        return results
    
    async def youtube_search(self, query: str, max_results: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for videos using YouTube Data API with duration and quality filters.
        Args:
            query (str): The search query.
            max_results (int): Maximum number of results to return.
        Returns:
            List[Dict[str, Any]]: List of video information dictionaries.
        """
        try:
            loop = asyncio.get_running_loop()
            # googleapiclient is not async, so run in executor
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
                # Get video details
                video_response = self.youtube.videos().list(
                    id=','.join(video_ids),
                    part='contentDetails,statistics,snippet,status,player'
                ).execute()
                videos = []
                for item in video_response.get('items', []):
                    duration_str = item['contentDetails']['duration']
                    duration_seconds = self._parse_duration(duration_str)
                    videos.append({
                        'video_id': item['id'],
                        'url': f'https://www.youtube.com/watch?v={item["id"]}',
                        'title': item['snippet']['title'],
                        'description': item['snippet'].get('description', ''),
                        'duration': duration_seconds,
                        'thumbnail': item['snippet'].get('thumbnails', {}).get('high', {}).get('url', ''),
                        'view_count': int(item['statistics'].get('viewCount', 0)),
                        'like_count': int(item['statistics'].get('likeCount', 0)),
                        'channel_title': item['snippet'].get('channelTitle', ''),
                        'published_at': item['snippet'].get('publishedAt', ''),
                        'category_id': item['snippet'].get('categoryId', ''),
                        'definition': item['contentDetails'].get('definition', '')
                    })
                return videos
            return await loop.run_in_executor(None, _search)
        except HttpError as e:
            print(f"YouTube API error: {e}")
            return []
        except Exception as e:
            print(f"Error searching YouTube content: {e}")
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
        """
        query = await self.query_builder.perform_action(user_query, query_type)
        query = query.get('query', user_query)
        if not query:
            print("No optimized query generated.")
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
