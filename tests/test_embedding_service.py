import pytest
from app.services.embedding_service import EmbeddingService
import asyncio

class DummyEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

class DummyEmbeddingService(EmbeddingService):
    def __init__(self):
        super().__init__()
        self.embedder = DummyEmbedder()

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
    service = DummyEmbeddingService()
    result = await service.perform_action(text="abc")
    assert result == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_perform_action_texts():
    service = DummyEmbeddingService()
    result = await service.perform_action(texts=["a", "b"])
    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
