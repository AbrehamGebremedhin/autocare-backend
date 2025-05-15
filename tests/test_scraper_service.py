import pytest
from app.services.scraper_service import ScraperService

class DummyDriver:
    def get(self, url):
        pass
    def quit(self):
        pass
    def execute_script(self, script):
        return None
    @property
    def title(self):
        return "Test Title"
    def find_element(self, by, value):
        class DummyElement:
            def get_attribute(self, attr):
                return "Test Meta"
            @property
            def text(self):
                return "Test Text"
        return DummyElement()

class DummyScraperService(ScraperService):
    def _get_driver(self):
        return DummyDriver()

import asyncio
@pytest.mark.asyncio
async def test_extract_page_info():
    service = DummyScraperService()
    result = await service.extract_page_info("http://example.com")
    assert result["title"] == "Test Title"
    assert result["text"] == "Test Text"
    assert result["meta_description"] == "Test Meta"
