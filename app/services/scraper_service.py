from app.services.base_service import BaseService
from typing import List, Dict, Any
import httpx
from bs4 import BeautifulSoup

class ScraperService(BaseService):
    """
    Service to scrape a list of URLs and extract page information (title, text, meta description).
    """
    def __init__(self):
        super().__init__()

    async def extract_page_info(self, url: str) -> Dict[str, Any]:
        """
        Fetches a page and extracts its title, text content, and meta description.
        Args:
            url (str): The URL to scrape.
        Returns:
            Dict[str, Any]: Extracted information.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string.strip() if soup.title and soup.title.string else ''
                meta_desc = ''
                meta = soup.find('meta', attrs={'name': 'description'})
                if meta and meta.get('content'):
                    meta_desc = meta['content'].strip()
                # Extract visible text (basic)
                for script in soup(['script', 'style']):
                    script.decompose()
                text = ' '.join(soup.stripped_strings)
                return {
                    'url': url,
                    'title': title,
                    'meta_description': meta_desc,
                    'text': text[:2000]  # Limit to first 2000 chars for brevity
                }
            except Exception as e:
                return {
                    'url': url,
                    'error': str(e)
                }

    async def perform_action(self, links: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape a list of links and extract page information.
        Args:
            links (List[str]): List of URLs to scrape.
            limit (int): Max number of links to process.
        Returns:
            List[Dict[str, Any]]: List of extracted page info dicts.
        """
        results = []
        for url in links[:limit]:
            info = await self.extract_page_info(url)
            results.append(info)
        return results
