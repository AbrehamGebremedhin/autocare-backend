from app.services.base_service import BaseService
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import random
import asyncio
from typing import Optional, List, Dict
from app.utils.logger import Logger

class ScraperService(BaseService):
    """
    Enhanced service to scrape URLs using Crawl4AI with async capabilities, retry logic, and performance optimizations.
    """
    def __init__(self, headless: bool = True, logger: Optional[Logger] = None, pool_size: int = 3, websocket_manager=None):
        """
        Initialize the ScraperService with crawl4ai configuration.
        Args:
            headless (bool): Whether to run the browser in headless mode.
            logger (Logger): Optional logger instance.
            pool_size (int): Number of concurrent crawlers (used for semaphore).
            websocket_manager: Optional WebSocketManager for notifications.
        """
        super().__init__(websocket_manager=websocket_manager)
        self.headless = headless
        self.logger = logger or Logger("ScraperService")
        self.pool_size = pool_size
        self._max_retries = 3
        self._retry_delay = 1.0
        self.crawler = None

    async def _get_crawler_config(self) -> dict:
        """
        Get crawler configuration for crawl4ai.
        Returns:
            dict: Crawler configuration options.
        """
        # Use realistic user agents
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0',
        ]
        
        return {
            "headless": self.headless,
            "browser_type": "chromium",
            "user_agent": random.choice(user_agents),
            "viewport_width": 1920,
            "viewport_height": 1080,
            "accept_downloads": False,
            "headers": {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.google.com/',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',                'DNT': '1',
            }
        }

    async def _get_crawl_config(self, url: str) -> CrawlerRunConfig:
        """
        Get crawl configuration for a specific URL.
        Args:
            url (str): The URL to crawl.
        Returns:
            CrawlerRunConfig: Configuration for the crawl operation.
        """
        return CrawlerRunConfig(
            word_count_threshold=10,
            extraction_strategy=None,  # Use default markdown extraction
            cache_mode="bypass",  # Use string instead of enum
            page_timeout=10000,  # 10 seconds
            delay_before_return_html=2.0,  # Wait 2 seconds for dynamic content
            remove_overlay_elements=True,
            screenshot=False,  # Disable screenshots for performance
            pdf=False,  # Disable PDF generation
        )

    async def cleanup(self):
        """
        Cleanup method to close the crawler.
        """
        if self.crawler:
            try:
                await self.crawler.close()
                self.crawler = None
            except Exception as e:
                await self.logger.error(f"Error during crawler cleanup: {e}")

    async def extract_page_info_with_retry(self, url: str) -> dict:
        """
        Fetch a page and extract its information with retry logic.
        Args:
            url (str): The URL to scrape.
        Returns:
            dict: Extracted information.
        """
        for attempt in range(self._max_retries):
            try:
                result = await self.extract_page_info(url)
                if 'error' not in result:
                    return result
                elif attempt == self._max_retries - 1:
                    return result
            except Exception as e:
                if attempt == self._max_retries - 1:
                    await self.logger.error(f"Final attempt failed for {url}: {e}")
                    return {'url': url, 'error': str(e)}
                
                wait_time = self._retry_delay * (2 ** attempt)
                await self.logger.warning(f"Attempt {attempt + 1} failed for {url}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
        
        return {'url': url, 'error': 'Max retries exceeded'}

    async def extract_page_info(self, url: str) -> dict:
        """
        Fetch a page and extract its title, text content, and meta description using Crawl4AI.
        Args:
            url (str): The URL to scrape.
        Returns:
            dict: Extracted information.
        """
        try:            # Initialize crawler if not already done
            if not self.crawler:
                crawler_config = await self._get_crawler_config()
                self.crawler = AsyncWebCrawler(**crawler_config)
                await self.crawler.start()            # Get crawl configuration
            crawl_config = await self._get_crawl_config(url)
            
            # Perform the crawl - pass URL as first argument and config separately
            result = await self.crawler.arun(url, config=crawl_config)
            
            if not result.success:
                error_msg = f"Crawl failed: {result.error_message}" if hasattr(result, 'error_message') else "Unknown crawl error"
                await self.logger.error(f"Failed to crawl {url}: {error_msg}")
                return {'url': url, 'error': error_msg}
            
            # Extract title - try multiple sources
            title = ""
            if hasattr(result, 'metadata') and result.metadata and result.metadata.get('title'):
                title = result.metadata['title']
            elif hasattr(result, 'extracted_content') and result.extracted_content:
                # Try to extract title from markdown content
                lines = result.markdown.split('\n')
                for line in lines:
                    if line.startswith('# '):
                        title = line[2:].strip()
                        break
            
            # Extract meta description
            meta_desc = ""
            if hasattr(result, 'metadata') and result.metadata:
                meta_desc = result.metadata.get('description', '') or result.metadata.get('og:description', '')
            
            # Get cleaned text content
            text_content = ""
            if hasattr(result, 'cleaned_html') and result.cleaned_html:
                # Use cleaned HTML if available, strip HTML tags for plain text
                import re
                text_content = re.sub(r'<[^>]+>', ' ', result.cleaned_html)
                text_content = ' '.join(text_content.split())  # Normalize whitespace
            elif hasattr(result, 'markdown') and result.markdown:
                # Use markdown content as fallback
                text_content = result.markdown
                # Remove markdown formatting for cleaner text
                import re
                text_content = re.sub(r'[#*`\[\]()_~]', '', text_content)
                text_content = ' '.join(text_content.split())  # Normalize whitespace
            
            return {
                'url': url,
                'title': title.strip(),
                'meta_description': meta_desc.strip(),
                'text': text_content[:10000]  # Limit text length for performance
            }
            
        except Exception as e:
            await self.logger.error(f"extract_page_info error for {url}: {e}")
            return {'url': url, 'error': str(e)}

    async def perform_action(self, links: list, limit: int = 2, concurrency: int = 3) -> list:
        """
        Scrape a list of links with optimized parallel processing and error handling.
        Args:
            links (list): List of URLs to scrape.
            limit (int): Max number of links to process.
            concurrency (int): Max number of concurrent scrapes.
        Returns:
            list: List of extracted page info dicts.
        """
        return await self.run_with_notification(self._perform_action_impl, links, limit, concurrency)

    async def _perform_action_impl(self, links: list, limit: int = 2, concurrency: int = 3) -> list:
        """
        Internal implementation of perform_action without notification.
        """
        if not links:
            return []
        
        # Apply rate limiting per service
        await self._rate_limit()
        
        # Use semaphore for controlled concurrency
        sem = asyncio.Semaphore(min(concurrency, self.pool_size))
        results = []

        async def scrape_with_semaphore(url):
            async with sem:
                # Random delay to avoid overwhelming servers
                delay = random.uniform(0.2, 0.8)
                await asyncio.sleep(delay)
                return await self.extract_page_info_with_retry(url)

        # Process links in batches to manage memory
        batch_size = 5
        for i in range(0, min(len(links), limit), batch_size):
            batch_links = links[i:i + batch_size]
            
            tasks = [scrape_with_semaphore(url) for url in batch_links]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    await self.logger.error(f"Batch processing error: {result}")
                    results.append({'url': 'unknown', 'error': str(result)})
                else:
                    results.append(result)
            
            # Small delay between batches
            if i + batch_size < min(len(links), limit):
                await asyncio.sleep(0.5)
        
        return results

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if self.crawler:
                asyncio.create_task(self.cleanup())
        except Exception:
            pass
