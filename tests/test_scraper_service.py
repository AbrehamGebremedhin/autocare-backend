import pytest
from app.services.scraper_service import ScraperService
from app.utils.logger import Logger

class DummyWebSocketManager:
    def __init__(self):
        self.messages = []
    async def broadcast(self, message):
        self.messages.append(message)

class DummyScraperService(ScraperService):
    def __init__(self, websocket_manager=None):
        super().__init__(websocket_manager=websocket_manager)
    async def _perform_action_impl(self, *args, **kwargs):
        return 'scraped'
    async def perform_action(self, *args, **kwargs):
        return await self.run_with_notification(self._perform_action_impl, *args, **kwargs)

@pytest.mark.asyncio
async def test_extract_page_info():
    """Test that scraper service can extract page info using crawl4ai."""
    logger = Logger("TestLogger")
    service = ScraperService(headless=True, logger=logger)
    
    try:
        # Test with a reliable, simple webpage
        result = await service.extract_page_info("https://example.com")
        
        # Verify the structure of the response
        assert "url" in result
        assert "title" in result
        assert "text" in result
        assert "meta_description" in result
        assert result["url"] == "https://example.com"
        
        # Verify that it's not an error response
        assert "error" not in result
        
        # Verify we got some content
        assert len(result["text"]) > 0
        
        # The actual title should be "Example Domain" for example.com
        assert result["title"] == "Example Domain"
        
    finally:
        # Ensure cleanup
        await service.cleanup()

@pytest.mark.asyncio
async def test_perform_action_multiple_urls():
    """Test that scraper service can handle multiple URLs."""
    logger = Logger("TestLogger")
    service = ScraperService(headless=True, logger=logger)
    
    try:
        urls = ["https://example.com"]
        results = await service.perform_action(urls, limit=1, concurrency=1)
        
        assert len(results) == 1
        assert "error" not in results[0]
        assert results[0]["title"] == "Example Domain"
        
    finally:
        await service.cleanup()

@pytest.mark.asyncio
async def test_extract_page_info_error_handling():
    """Test error handling for invalid URLs."""
    logger = Logger("TestLogger")
    service = ScraperService(headless=True, logger=logger)
    
    try:
        # Test with an invalid URL
        result = await service.extract_page_info("https://this-domain-definitely-does-not-exist-12345.com")
        
        # Should return an error response
        assert "error" in result
        assert result["url"] == "https://this-domain-definitely-does-not-exist-12345.com"
        
    finally:
        await service.cleanup()

@pytest.mark.asyncio
async def test_perform_action_with_notification():
    ws = DummyWebSocketManager()
    service = DummyScraperService(websocket_manager=ws)
    result = await service.perform_action()
    assert result == 'scraped'
    assert ws.messages[0]['event'] == 'task_started'
    assert ws.messages[-1]['event'] == 'task_finished'
