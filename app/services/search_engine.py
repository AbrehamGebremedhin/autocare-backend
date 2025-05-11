from app.services.base import BaseService
from googlesearch import search
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from app.core.config import get_settings
from app.services.query_builder import QueryBuilderService
from typing import List, Dict, Any

class SearchEngineService(BaseService):
    """
    Service to perform searches using optimized queries for vector search, YouTube, and search engines.
    Provides unified access to web, YouTube, and vector search functionalities.
    """
    def __init__(self):
        """
        Initialize the SearchEngineService with required dependencies and API clients.
        """
        super().__init__()
        settings = get_settings()
        self.query_builder = QueryBuilderService()
        self.youtube = build('youtube', 'v3', developerKey=settings.YOUTUBE_API_KEY)

    def vector_search(self, query: str):
        """
        Placeholder for vector search implementation.
        Args:
            query (str): The search query.
        Returns:
            List of search results (to be implemented).
        """
        pass

    def web_search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
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
            for result in search(query, num_results=num_results, advanced=True):
                results.append({
                    'title': result.title,
                    'link': result.url,
                    'description': result.description
                })
        except Exception as e:
            print(f"Error during Google search: {e}")
        return results
    
    def youtube_search(self, query: str, max_results: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for videos using YouTube Data API with duration and quality filters.
        Args:
            query (str): The search query.
            max_results (int): Maximum number of results to return.
        Returns:
            List[Dict[str, Any]]: List of video information dictionaries.
        """
        try:
            # Search for videos
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

    def perform_action(self, user_query: str, query_type: str = None):
        """
        Perform a search action based on the query type.
        Args:
            user_query (str): The user's search query.
            query_type (str, optional): The type of search ('search_engine', 'youtube', 'vector').
        Returns:
            List of search results from the selected search method.
        """
        query = self.query_builder.perform_action(user_query, query_type)
        query = query.get('query', user_query)
        if not query:
            print("No optimized query generated.")
            return []
        print(f"Optimized query: {query}")
        if not query_type:
            return self.web_search(query)
        query_type = query_type.lower()
        if query_type == "search_engine":
            return self.web_search(query)
        elif query_type == "youtube":
            return self.youtube_search(query)
        elif query_type == "vector":
            return self.vector_search(query)
        else:
            return self.web_search(user_query)
from pprint import pprint
# Usage example:
search_service = SearchEngineService()
results = search_service.perform_action("how to fix engine overheating in toyota echo 2001", "search_engine")
pprint(results)
