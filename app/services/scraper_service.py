from app.services.base_service import BaseService
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
import random
import time
import asyncio
from typing import Optional
from app.utils.logger import Logger

class ScraperService(BaseService):
    """
    Service to scrape a list of URLs and extract page information (title, text, meta description) using Selenium.
    """
    def __init__(self, headless: bool = True, logger: Optional[Logger] = None):
        """
        Initialize the ScraperService.
        Args:
            headless (bool): Whether to run the browser in headless mode.
            logger (Logger): Optional logger instance.
        """
        super().__init__()
        self.headless = headless
        self.logger = logger or Logger("ScraperService")

    def _get_headers(self) -> dict:
        """
        Generate headers to mimic a real browser and avoid detection.
        Returns:
            dict: HTTP headers for requests.
        """
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0',
        ]
        return {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive',
        }

    def _get_driver(self):
        """
        Create and configure a Selenium Chrome WebDriver instance.
        Returns:
            WebDriver: Configured Chrome WebDriver.
        """
        options = Options()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('user-agent=' + self._get_headers()['User-Agent'])
        driver = webdriver.Chrome(options=options)
        return driver

    async def extract_page_info(self, url: str) -> dict:
        """
        Fetch a page and extract its title, text content, and meta description using Selenium.
        Args:
            url (str): The URL to scrape.
        Returns:
            dict: Extracted information.
        """
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._extract_page_info_sync(url))
        except Exception as e:
            await self.logger.error(f"extract_page_info error: {e}")
            return {'url': url, 'error': str(e)}

    def _extract_page_info_sync(self, url: str) -> dict:
        """
        Synchronous helper to extract page info using Selenium.
        Args:
            url (str): The URL to scrape.
        Returns:
            dict: Extracted information.
        """
        driver = self._get_driver()
        try:
            driver.get(url)
            time.sleep(2)  # Wait for page to load
            title = driver.title.strip() if driver.title else ''
            meta_desc = ''
            try:
                meta = driver.find_element(By.XPATH, '//meta[@name="description"]')
                meta_desc = meta.get_attribute('content') or ''
            except Exception:
                pass
            # Remove script/style and get visible text
            driver.execute_script('''
                var scripts = document.querySelectorAll('script, style');
                scripts.forEach(function(s) { s.parentNode.removeChild(s); });
            ''')
            body = driver.find_element(By.TAG_NAME, 'body')
            text = body.text.strip().replace('\n', ' ')
            return {
                'url': url,
                'title': title,
                'meta_description': meta_desc.strip(),
                'text': text
            }
        except WebDriverException as e:
            if hasattr(self, 'logger'):
                asyncio.run(self.logger.error(f"_extract_page_info_sync WebDriverException: {e}"))
            return {'url': url, 'error': str(e)}
        except Exception as e:
            if hasattr(self, 'logger'):
                asyncio.run(self.logger.error(f"_extract_page_info_sync error: {e}"))
            return {'url': url, 'error': str(e)}
        finally:
            driver.quit()

    async def perform_action(self, links: list, limit: int = 10) -> list:
        """
        Scrape a list of links and extract page information.
        Args:
            links (list): List of URLs to scrape.
            limit (int): Max number of links to process.
        Returns:
            list: List of extracted page info dicts.
        """
        results = []
        for url in links[:limit]:
            delay = random.uniform(1, 3)
            await asyncio.sleep(delay)
            info = await self.extract_page_info(url)
            results.append(info)
        return results
