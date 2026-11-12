import httpx
from urllib.parse import quote

from bs4 import BeautifulSoup


class FetchCarDataService:
    """startmycar.com: the page that links a car's owner manual. The PDF itself is downloaded by manual_search.fetch_pdf."""

    BASE_URL = "https://www.startmycar.com/"

    def __init__(self):
        self._client_pool: httpx.AsyncClient | None = None

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

    def build_url(self, make: str, model: str, year: int) -> dict:
        """The owner-manual page URL for make, model and year ({"Owner_Manual": url}). ValueError if any is missing."""
        if not make or not model or not year:
            raise ValueError("Make, model, and year must be provided.")

        # Make and model come from the user: quote them so they stay one path segment each.
        make = quote(make.replace(" ", "-").lower(), safe="")
        model = quote(model.replace(" ", "-").lower(), safe="")
        return {"Owner_Manual": f"{self.BASE_URL}{make}/{model}/info/manuals/{year}"}

    async def scrape_links(self, url: str) -> str:
        """The first link in the page's owner-manual block, or "" when it has none. Raises on an error status."""
        response = await (await self._get_client()).get(url)
        response.raise_for_status()
        manual_div = BeautifulSoup(response.text, 'html.parser').find('div', class_='manual-text')
        first_link = manual_div.find('a', href=True) if manual_div else None
        return first_link['href'] if first_link else ""
