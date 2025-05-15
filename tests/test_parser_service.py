import pytest
from app.services.parser_service import ParserService
import asyncio

class DummyPdfReader:
    def __init__(self, file_path):
        self.pages = [self]
    def extract_text(self):
        return "This is a test PDF."

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
