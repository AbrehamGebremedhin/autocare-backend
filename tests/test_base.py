import pytest
from unittest.mock import patch, MagicMock
from app.db.base import SupabaseDBHandler, get_db_handler

# Add your tests for base here
def test_placeholder():
    assert True

@patch('app.db.base.create_client', return_value=MagicMock())
@patch('app.db.base.get_settings')
def test_supabase_db_handler_singleton(mock_settings, mock_create):
    mock_settings.return_value.SUPABASE_URL = 'url'
    mock_settings.return_value.SUPABASE_KEY = 'key'
    handler1 = SupabaseDBHandler()
    handler2 = SupabaseDBHandler()
    assert handler1 is handler2
    assert handler1._client is not None

@patch('app.db.base.create_client', return_value=MagicMock())
@patch('app.db.base.get_settings')
def test_supabase_db_handler_missing_env(mock_settings, mock_create):
    mock_settings.return_value.SUPABASE_URL = ''
    mock_settings.return_value.SUPABASE_KEY = ''
    SupabaseDBHandler._instance = None
    with pytest.raises(ValueError):
        SupabaseDBHandler()

@pytest.mark.asyncio
async def test_get_db_handler():
    handler = await get_db_handler()
    assert isinstance(handler, SupabaseDBHandler)
