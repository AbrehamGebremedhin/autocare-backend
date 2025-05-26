from app.services.base_service import BaseService
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
import time
import asyncio
from typing import Optional, List, Dict
from app.utils.logger import Logger
import threading
from queue import Queue, Empty

class ScraperService(BaseService):
    """
    Enhanced service to scrape URLs with connection pooling, retry logic, and performance optimizations.
    """
    def __init__(self, headless: bool = True, logger: Optional[Logger] = None, pool_size: int = 3):
        """
        Initialize the ScraperService with connection pooling.
        Args:
            headless (bool): Whether to run the browser in headless mode.
            logger (Logger): Optional logger instance.
            pool_size (int): Number of browser instances in pool.
        """
        super().__init__()
        self.headless = headless
        self.logger = logger or Logger("ScraperService")
        self.pool_size = pool_size
        self.driver_pool = Queue(maxsize=pool_size)
        self.pool_lock = threading.Lock()
        self._initialize_pool()
        self._max_retries = 3
        self._retry_delay = 1.0

    def _initialize_pool(self):
        """Initialize the driver pool with configured browsers."""
        for _ in range(self.pool_size):
            driver = self._create_driver()
            self.driver_pool.put(driver)

    def _get_headers(self) -> dict:
        """
        Generate realistic headers to avoid detection.
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
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1',
        }

    def _create_driver(self):
        """
        Create and configure a Selenium Chrome WebDriver instance.
        Returns:
            WebDriver: Configured Chrome WebDriver.
        """
        options = Options()
        if self.headless:
            options.add_argument('--headless=new')  # Use new headless mode
        
        # Performance optimizations
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--disable-images')  # Don't load images for faster loading
        options.add_argument('--disable-javascript')  # Disable JS for simple text extraction
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-web-security')
        options.add_argument('--allow-running-insecure-content')
        
        # Set user agent
        options.add_argument(f'user-agent={self._get_headers()["User-Agent"]}')
        
        # Additional performance settings
        prefs = {
            "profile.default_content_setting_values": {
                "images": 2,  # Block images
                "plugins": 2,  # Block plugins
                "popups": 2,  # Block popups
                "geolocation": 2,  # Block location sharing
                "notifications": 2,  # Block notifications
                "media_stream": 2,  # Block media stream
            },
            "profile.managed_default_content_settings": {
                "images": 2
            }
        }
        options.add_experimental_option("prefs", prefs)
        
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(10)  # 10 second timeout
            driver.implicitly_wait(3)  # 3 second implicit wait
            return driver
        except Exception as e:
            if hasattr(self, 'logger'):
                asyncio.run(self.logger.error(f"Failed to create driver: {e}"))
            raise

    def _get_driver(self):
        """Get a driver from the pool."""
        try:
            return self.driver_pool.get_nowait()
        except Empty:
            # If pool is empty, create a new driver temporarily
            return self._create_driver()

    def _return_driver(self, driver):
        """Return a driver to the pool."""
        try:
            if driver and not self.driver_pool.full():
                self.driver_pool.put_nowait(driver)
            elif driver:
                # Pool is full, close the extra driver
                driver.quit()
        except Exception:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    def cleanup(self):
        """
        Cleanup method to close all browser instances in the pool.
        """
        while not self.driver_pool.empty():
            try:
                driver = self.driver_pool.get_nowait()
                driver.quit()
            except Exception:
                pass

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
        Optimized synchronous helper to extract page info using Selenium.
        Args:
            url (str): The URL to scrape.
        Returns:
            dict: Extracted information.
        """
        driver = None
        try:
            driver = self._get_driver()
            
            # Navigate to URL with timeout
            driver.get(url)
            
            # Wait for page to load with explicit wait
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                pass  # Continue anyway
            
            # Extract title
            title = driver.title.strip() if driver.title else ''
            
            # Extract meta description
            meta_desc = ''
            try:
                meta_elements = driver.find_elements(By.XPATH, '//meta[@name="description" or @property="og:description"]')
                if meta_elements:
                    meta_desc = meta_elements[0].get_attribute('content') or ''
            except Exception:
                pass
            
            # Remove script/style elements and extract text efficiently
            try:
                driver.execute_script('''
                    var elements = document.querySelectorAll('script, style, nav, header, footer, aside, .advertisement, .ads, .sidebar');
                    elements.forEach(function(el) { 
                        if (el.parentNode) el.parentNode.removeChild(el); 
                    });
                ''')
                
                # Get main content areas first
                content_selectors = [
                    'main', 'article', '.content', '.main-content', 
                    '.post-content', '.entry-content', '.article-content'
                ]
                
                text_content = ''
                for selector in content_selectors:
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            text_content = elements[0].text.strip()
                            break
                    except Exception:
                        continue
                
                # Fallback to body if no main content found
                if not text_content:
                    body = driver.find_element(By.TAG_NAME, 'body')
                    text_content = body.text.strip()
                
                # Clean up text
                text_content = ' '.join(text_content.split())  # Normalize whitespace
                
            except Exception as e:
                text_content = f'Error extracting content: {str(e)}'
            
            return {
                'url': url,
                'title': title,
                'meta_description': meta_desc.strip(),
                'text': text_content[:10000]  # Limit text length for performance
            }
            
        except WebDriverException as e:
            return {'url': url, 'error': f'WebDriver error: {str(e)}'}
        except Exception as e:
            return {'url': url, 'error': f'Extraction error: {str(e)}'}
        finally:
            if driver:
                self._return_driver(driver)

    async def perform_action(self, links: list, limit: int = 2, concurrency: int = 3) -> list:
        """
        Scrape a list of links with optimized parallel processing and error handling.
        Args:
            links (list): List of URLs to scrape.
            limit (int): Max number of links to process.
            concurrency (int): Max number of concurrent scrapes (reduced for stability).
        Returns:
            list: List of extracted page info dicts.
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
            self.cleanup()
        except Exception:
            pass
