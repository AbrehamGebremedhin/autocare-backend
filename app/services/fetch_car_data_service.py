from app.services.base_service import BaseService
from app.core.config import get_settings
from bs4 import BeautifulSoup
import httpx
import asyncio

class FetchCarDataService(BaseService):
    """
    Service to download a PDF file from a given URL and optionally notify via websocket.
    """
    def __init__(self, websocket_manager=None):
        """
        Initializes the FetchCarDataService with an optional websocket manager.
        Args:
            websocket_manager (WebSocketManager, optional): An instance of WebSocketManager for sending notifications.
        """
        super().__init__()
        self.websocket_manager = websocket_manager
        settings = get_settings()
        self.BASE_URL = settings.BASE_URL

    async def notify_websocket(self, message):
        """
        Sends a websocket notification message.
        Handles both async and sync contexts.
        Args:
            message (str): The message to send via websocket.
        """
        if self.websocket_manager is None:
            return  # No websocket manager, skip notification
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.websocket_manager.broadcast(message))
        except RuntimeError:
            await self.websocket_manager.broadcast(message)

    def build_url(self, make: str, model:str, year: int) -> str:
        """
        Constructs a URL for fetching car data based on make, model, and year.
        Args:
            make (str): The car's make.
            model (str): The car's model.
            year (int): The car's year.
        Returns:
            str: The constructed URL.
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

    async def scrape_links(self, url: str, req_type: str) -> list:
        """
        Scrapes all links from a given URL using BeautifulSoup.
        
        Args:
            url (str): The URL to scrape for links.
            
        Returns:
            list: A list of dictionaries containing link information.
        """
        async with httpx.AsyncClient() as client:
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
                raise Exception(f"Error scraping links: {str(e)}")

    async def download_pdf(self, url: str, output_path: str) -> None:
        """
        Downloads a PDF file from the given URL and saves it to the specified output path in the background.
        Optionally sends a websocket message on success or failure.
        Args:
            url (str): The URL of the PDF file to download.
            output_path (str): The local file path to save the downloaded PDF.
        Raises:
            Exception: If the download fails or the response is not a PDF.
        """
        async def _download():
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(url)
                    if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
                        with open(output_path, 'wb') as f:
                            f.write(response.content)
                        await self.notify_websocket(f"PDF downloaded successfully: {output_path}")
                    else:
                        await self.notify_websocket(f"Failed to download PDF from {url}")
                except Exception as e:
                    await self.notify_websocket(f"Error downloading PDF from {url}: {str(e)}")
        # Run the download in the background
        asyncio.create_task(_download())

    async def perform_action(self, make: str, model: str, year: int):
        """
        Performs the complete car data fetching action: building URLs, scraping links, and downloading PDFs.
        
        Args:
            make (str): The car's make.
            model (str): The car's model.
            year (int): The car's year.
            output_path (str, optional): Path to save downloaded files.
            req_type (str, optional): Type of request ("owner_manual" or "Car_guide_link").
            
        Returns:
            list: A list of scraped links or file paths.
        """
        links = self.build_url(make, model, year)
        
        await self.notify_websocket(f"Fetching data for {make} {model} {year}")

        manual_link = await self.scrape_links(links['Owner_Manual'], req_type="owner_manual")
        self.download_pdf(manual_link, output_path=f"{make}-{model}_{year}_EN_US.pdf")

        return await self.scrape_links(links['Car_guide_link'], req_type="car_guide_link")
