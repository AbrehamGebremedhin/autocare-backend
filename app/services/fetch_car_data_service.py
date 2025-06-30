from app.services.base_service import BaseService
from app.core.config import get_settings
from app.core.interfaces import IWebSocketManager, ILogger
from bs4 import BeautifulSoup
import httpx
import asyncio
from typing import Optional, List, Dict
from app.utils.logger import get_logger_instance, Logger
from app.utils.message_types import MessageSource

class FetchCarDataService(BaseService):
    """
    Enhanced service to download files and scrape car data with connection pooling and performance optimizations.
    """
    def __init__(
        self,
        websocket_manager: IWebSocketManager = None,
        logger: Optional[ILogger] = None,
        client_pool: Optional[httpx.AsyncClient] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the FetchCarDataService.
        Args:
            websocket_manager: Optional WebSocketManager for notifications.
            logger: Optional logger instance.
            client_pool: Optional HTTP client pool.
            base_url: Optional base URL for car data.
            max_retries: Max retries for HTTP requests.
            retry_delay: Delay between retries.
        """
        super().__init__(websocket_manager=websocket_manager)
        self.logger = logger or get_logger_instance("FetchCarDataService")
        settings = get_settings()
        self.BASE_URL = base_url or settings.BASE_URL
        self._client_pool = client_pool
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._client_pool is None:
            self._client_pool = httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
            )
        return self._client_pool

    async def cleanup(self):
        """Clean up HTTP client connections."""
        if self._client_pool:
            await self._client_pool.aclose()
            self._client_pool = None

    async def notify_websocket(self, message):
        """
        Sends a websocket notification message.
        Handles both async and sync contexts.
        Args:
            message (str): The message to send via websocket.
        """
        if self.websocket_manager is None:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.websocket_manager.broadcast(message))
        except RuntimeError:
            await self.websocket_manager.broadcast(message)
        except Exception as e:
            await self.logger.error(f"notify_websocket error: {e}")

    def build_url(self, make: str, model: str, year: int) -> dict:
        """
        Constructs a URL for fetching car data based on make, model, and year.
        Args:
            make (str): The car's make.
            model (str): The car's model.
            year (int): The car's year.
        Returns:
            dict: The constructed URLs for manuals and guides.
        Raises:
            ValueError: If any argument is missing.
        """
        if not make or not model or not year:
            raise ValueError("Make, model, and year must be provided.")

        make = make.replace(" ", "-").lower()
        model = model.replace(" ", "-").lower()
        year = str(year).lower()
        data = {
            "Owner_Manual": f"{self.BASE_URL}{make}/{model}/info/manuals/{year}",
            "Car_guide_link": f"{self.BASE_URL}{make}/{model}/guides/"
        }
        return data

    async def scrape_links_with_retry(self, url: str, req_type: str) -> list:
        """
        Scrape links with retry logic and improved error handling.
        Args:
            url (str): The URL to scrape for links.
            req_type (str): The type of request ('owner_manual' or 'car_guide_link').
        Returns:
            list: A list of scraped links or file paths.
        """
        for attempt in range(self._max_retries):
            try:
                result = await self.scrape_links(url, req_type)
                return result
            except Exception as e:
                if attempt == self._max_retries - 1:
                    await self.logger.error(f"Final attempt failed for {url}: {e}")
                    raise e
                
                wait_time = self._retry_delay * (2 ** attempt)
                await self.logger.warning(f"Attempt {attempt + 1} failed for {url}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
        
        return []

    async def scrape_links(self, url: str, req_type: str) -> list:
        """
        Scrapes all links from a given URL using BeautifulSoup with optimized parsing.
        Args:
            url (str): The URL to scrape for links.
            req_type (str): The type of request ('owner_manual' or 'car_guide_link').
        Returns:
            list: A list of scraped links or file paths.
        Raises:
            Exception: If scraping fails.
        """
        client = await self._get_client()
        await self._rate_limit()
        
        try:
            if req_type == "owner_manual":
                response = await client.get(url)
                
                if response.status_code != 200:
                    await self.notify_websocket(f"Failed to access {url}. Status code: {response.status_code}")
                    raise Exception(f"Failed to access URL. Status code: {response.status_code}")
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                manual_div = soup.find('div', class_='manual-text')
                link = ""
                if manual_div:
                    first_link = manual_div.find('a', href=True)
                    if first_link:
                        link = first_link['href']
                        
                await self.notify_websocket(f"Successfully scraped link from manual-text div at {url}")
                return link
            
            else:
                response = await client.get(url)
                                    
                if response.status_code != 200:
                    await self.notify_websocket(f"Failed to access {url}. Status code: {response.status_code}")
                    raise Exception(f"Failed to access URL. Status code: {response.status_code}")
                    
                soup = BeautifulSoup(response.text, 'html.parser')

                links = []
                article_links = soup.find_all('a', class_='ArticleLink')
                for link in article_links:
                    if link.has_attr('href'):
                        link_info = f"{self.BASE_URL}{link['href'].lstrip('/')}"
                        links.append(link_info)
                                        
                await self.notify_websocket(f"Successfully scraped {len(links)} article links from {url}")
                return links
                
        except Exception as e:
            await self.notify_websocket(f"Error scraping links from {url}: {str(e)}")
            await self.logger.error(f"scrape_links error: {e}")
            raise

    async def download_pdf(self, url: str, output_path: str) -> None:
        """
        Downloads a PDF file from the given URL and saves it to the specified output path.
        Enhanced with connection pooling and retry logic.
        Args:
            url (str): The URL of the PDF file to download.
            output_path (str): The local file path to save the downloaded PDF.
        """
        for attempt in range(self._max_retries):
            try:
                client = await self._get_client()
                await self._rate_limit()
                
                response = await client.get(url)
                if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    await self.notify_websocket(f"PDF downloaded successfully: {output_path}")
                    return
                else:
                    await self.notify_websocket(f"Failed to download PDF from {url}")
                    if attempt == self._max_retries - 1:
                        raise Exception(f"Failed to download PDF. Status: {response.status_code}")
                        
            except Exception as e:
                if attempt == self._max_retries - 1:
                    await self.notify_websocket(f"Error downloading PDF from {url}: {str(e)}")
                    await self.logger.error(f"download_pdf error: {e}")
                    raise
                
                wait_time = self._retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)

    async def perform_action(self, make: str, model: str, year: int, websocket=None, session_id=None):
        """
        Performs the complete car data fetching action: building URLs, scraping links, and downloading PDFs.
        Enhanced with parallel processing for improved performance.
        Args:
            make (str): The car's make.
            model (str): The car's model.
            year (int): The car's year.
        Returns:
            list: A list of scraped links or file paths.
        Raises:
            Exception: If any step fails.
        """
        if websocket:
            await self.send_ws_stage(websocket, "Car data fetch started", MessageSource.CHAT_SERVICE, session_id=session_id, details={"make": make, "model": model, "year": year})
        try:
            result = await self.run_with_notification(self._perform_action_impl, make, model, year)
            if websocket:
                await self.send_ws_result(websocket, "Car data fetch complete", MessageSource.CHAT_SERVICE, session_id=session_id, details={"num_links": len(result) if result else 0})
            return result
        except Exception as e:
            if websocket:
                await self.send_ws_error(websocket, f"FetchCarDataService.perform_action error: {e}", MessageSource.CHAT_SERVICE, session_id=session_id, details={"error": str(e)})
            raise

    async def _perform_action_impl(self, make: str, model: str, year: int):
        try:
            links = self.build_url(make, model, year)
            
            await self.notify_websocket(f"Fetching data for {make} {model} {year}")

            async def process_manual():
                manual_link = await self.scrape_links_with_retry(links['Owner_Manual'], req_type="owner_manual")
                if manual_link:
                    await self.download_pdf(manual_link, output_path=f"{make}-{model}_{year}_EN_US.pdf")
                return manual_link

            async def process_guides():
                return await self.scrape_links_with_retry(links['Car_guide_link'], req_type="car_guide_link")

            manual_task = asyncio.create_task(process_manual())
            guides_task = asyncio.create_task(process_guides())
            
            manual_result, guide_links = await asyncio.gather(manual_task, guides_task, return_exceptions=True)
            
            if isinstance(manual_result, Exception):
                await self.logger.error(f"Manual processing failed: {manual_result}")
                
            if isinstance(guide_links, Exception):
                await self.logger.error(f"Guide processing failed: {guide_links}")
                guide_links = []
            
            return guide_links if not isinstance(guide_links, Exception) else []
            
        except Exception as e:
            await self.logger.error(f"perform_action error: {e}")
            raise
