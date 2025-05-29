import pytest
from app.services.embedding_service import EmbeddingService
import asyncio

class DummyWebSocketManager:
    def __init__(self):
        self.messages = []
    async def broadcast(self, message):
        self.messages.append(message)

class DummyEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

class DummyEmbeddingService(EmbeddingService):
    def __init__(self, websocket_manager=None):
        super().__init__(websocket_manager=websocket_manager)
        self.embedder = DummyEmbedder()
    async def _perform_action_impl(self, *args, **kwargs):
        if 'text' in kwargs:
            return await self.embed_text(kwargs['text'])
        if 'texts' in kwargs:
            return await self.embed_texts(kwargs['texts'])
    async def perform_action(self, *args, **kwargs):
        return await self.run_with_notification(self._perform_action_impl, *args, **kwargs)

@pytest.mark.asyncio
async def test_embed_text():
    service = DummyEmbeddingService()
    result = await service.embed_text("test")
    assert result == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_embed_texts():
    service = DummyEmbeddingService()
    result = await service.embed_texts(["a", "b"])
    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

@pytest.mark.asyncio
async def test_perform_action_text():
    ws = DummyWebSocketManager()
    service = DummyEmbeddingService(websocket_manager=ws)
    result = await service.perform_action(text="abc")
    assert result == [0.1, 0.2, 0.3]
    assert ws.messages[0]['event'] == 'task_started'
    assert ws.messages[-1]['event'] == 'task_finished'

@pytest.mark.asyncio
async def test_perform_action_texts():
    ws = DummyWebSocketManager()
    service = DummyEmbeddingService(websocket_manager=ws)
    result = await service.perform_action(texts=["a", "b"])
    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert ws.messages[0]['event'] == 'task_started'
    assert ws.messages[-1]['event'] == 'task_finished'
