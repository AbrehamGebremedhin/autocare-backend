import pytest
from app.services.parser_service import ParserService
import asyncio

class DummyPdfReader:
    def __init__(self, file_path):
        self.pages = [self]
    def extract_text(self):
        return "This is a test PDF."

class DummyWebSocketManager:
    def __init__(self):
        self.messages = []
    async def broadcast(self, message):
        self.messages.append(message)

class DummyParserService(ParserService):
    async def _perform_action_impl(self, *args, **kwargs):
        return 'parsed'
    async def perform_action(self, *args, **kwargs):
        return await self.run_with_notification(self._perform_action_impl, *args, **kwargs)

@pytest.mark.asyncio
async def test_parse_string():
    service = ParserService()
    text = "a" * 2500
    chunks = await service.parse_string(text, chunk_size=1000)
    # Check that all chunks are not empty and reconstruct the original text
    assert all(isinstance(chunk, str) and chunk for chunk in chunks)
    assert ''.join(chunks) == text

@pytest.mark.asyncio
async def test_chunk_text():
    service = ParserService()
    text = "Sentence one. Sentence two. Sentence three."
    chunks = await service.chunk_text(text, chunk_size=10)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)

@pytest.mark.asyncio
async def test_perform_action_with_notification():
    ws = DummyWebSocketManager()
    service = DummyParserService(websocket_manager=ws)
    result = await service.perform_action()
    assert result == 'parsed'
    assert ws.messages[0]['event'] == 'task_started'
    assert ws.messages[-1]['event'] == 'task_finished'
