import pytest
from unittest.mock import patch, MagicMock
from app.utils.logger import Logger
import asyncio

# Add your tests for logger here
def test_placeholder():
    assert True

@patch('app.utils.logger.logging.getLogger')
def test_logger_singleton(mock_get_logger):
    logger1 = Logger('test')
    logger2 = Logger('test')
    assert logger1 is logger2

@pytest.mark.asyncio
async def test_logger_info():
    logger = Logger('test')
    with patch.object(logger.logger, 'info') as mock_info:
        await logger.info('msg')
        mock_info.assert_called()

@pytest.mark.asyncio
async def test_logger_warning():
    logger = Logger('test')
    with patch.object(logger.logger, 'warning') as mock_warning:
        await logger.warning('msg')
        mock_warning.assert_called()

@pytest.mark.asyncio
async def test_logger_error():
    logger = Logger('test')
    with patch.object(logger.logger, 'error') as mock_error:
        await logger.error('msg')
        mock_error.assert_called()

@pytest.mark.asyncio
async def test_logger_debug():
    logger = Logger('test')
    with patch.object(logger.logger, 'debug') as mock_debug:
        await logger.debug('msg')
        mock_debug.assert_called()
