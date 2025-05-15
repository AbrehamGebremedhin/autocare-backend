import pytest
from app.services.query_builder_service import QueryBuilderService

class DummyLLM:
    async def ainvoke(self, input):
        return "optimized query"

class DummyQueryBuilderService(QueryBuilderService):
    def __init__(self):
        super().__init__()
        self.llm = DummyLLM()

    async def build_search_engine_query(self, query):
        return "optimized query"

    async def build_youtube_query(self, query):
        return "optimized query"

@pytest.mark.asyncio
async def test_build_search_engine_query():
    service = DummyQueryBuilderService()
    result = await service.build_search_engine_query("test query")
    assert result == "optimized query"

@pytest.mark.asyncio
async def test_build_youtube_query():
    service = DummyQueryBuilderService()
    result = await service.build_youtube_query("test query")
    assert result == "optimized query"
