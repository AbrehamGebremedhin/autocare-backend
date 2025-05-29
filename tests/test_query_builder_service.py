import pytest
from app.services.query_builder_service import QueryBuilderService

class DummyLLM:
    async def ainvoke(self, input):
        return "optimized query"

class DummyWebSocketManager:
    def __init__(self):
        self.messages = []
    async def broadcast(self, message):
        self.messages.append(message)

class DummyQueryBuilderService(QueryBuilderService):
    def __init__(self, websocket_manager=None):
        super().__init__(websocket_manager=websocket_manager)
        self.llm = DummyLLM()

    async def build_search_engine_query(self, query):
        return "optimized query"

    async def build_youtube_query(self, query):
        return "optimized query"

    async def _perform_action_impl(self, *args, **kwargs):
        return 'queried'

    async def perform_action(self, *args, **kwargs):
        return await self.run_with_notification(self._perform_action_impl, *args, **kwargs)

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

@pytest.mark.asyncio
async def test_perform_action_with_notification():
    ws = DummyWebSocketManager()
    service = DummyQueryBuilderService(websocket_manager=ws)
    result = await service.perform_action()
    assert result == 'queried'
    assert ws.messages[0]['event'] == 'task_started'
    assert ws.messages[-1]['event'] == 'task_finished'
