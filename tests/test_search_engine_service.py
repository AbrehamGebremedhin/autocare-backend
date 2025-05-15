import pytest
from app.services.search_engine_service import SearchEngineService

class DummyQueryBuilder:
    async def build_search_engine_query(self, query):
        return "optimized query"
    async def build_youtube_query(self, query):
        return "youtube query"

class DummyYouTube:
    pass

class DummyEmbeddingService:
    pass

class DummyParserService:
    pass

class DummyScraperService:
    pass

class DummyBucketManager:
    pass

class DummyFetchCarDataService:
    pass

class DummyDBHandler:
    class _client:
        @staticmethod
        def table(name):
            class DummyTable:
                def select(self, *args, **kwargs):
                    class DummySelect:
                        def limit(self, n):
                            class DummyExecute:
                                data = [{"id": 1, "text": "test"}]
                                def execute(self):
                                    return self
                            return DummyExecute()
                    return DummySelect()
            return DummyTable()

@pytest.mark.asyncio
async def test_get_ground_knowledge_chunks():
    service = SearchEngineService(
        query_builder=DummyQueryBuilder(),
        youtube_client=DummyYouTube(),
        embedding_service=DummyEmbeddingService(),
        parser_service=DummyParserService(),
        scraper_service=DummyScraperService(),
        bucket_manager=DummyBucketManager(),
        fetch_car_data_service=DummyFetchCarDataService(),
        db_handler=DummyDBHandler()
    )
    result = await service._get_ground_knowledge_chunks(limit=1)
    assert isinstance(result, list)
    assert result[0]["id"] == 1
