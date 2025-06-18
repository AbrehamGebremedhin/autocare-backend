import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from app.utils import dataloader

# Add your tests for dataloader here
def test_placeholder():
    assert True

@pytest.mark.asyncio
async def test_process_pdf_success():
    parser = MagicMock()
    parser.parse_pdf = AsyncMock(return_value=['chunk1', 'chunk2'])
    embed = MagicMock()
    embed.embed_texts = AsyncMock(return_value=[[0.1]*768, [0.2]*768])
    milvus = MagicMock()
    milvus.insert = MagicMock()
    logger = MagicMock()
    logger.info = AsyncMock()
    logger.warning = AsyncMock()
    logger.error = AsyncMock()
    await dataloader.process_pdf('file.pdf', parser, embed, milvus, logger)
    assert logger.info.await_count >= 1

@pytest.mark.asyncio
async def test_process_pdf_no_chunks():
    parser = MagicMock()
    parser.parse_pdf = AsyncMock(return_value=[])
    embed = MagicMock()
    milvus = MagicMock()
    logger = MagicMock()
    logger.info = AsyncMock()
    logger.warning = AsyncMock()
    logger.error = AsyncMock()
    await dataloader.process_pdf('file.pdf', parser, embed, milvus, logger)
    logger.warning.assert_awaited()
