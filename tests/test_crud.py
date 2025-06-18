import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.db.crud import BaseCRUD

class DummyCRUD(BaseCRUD):
    async def unique_logic(self, *args, **kwargs):
        return 'unique'

@pytest.fixture
def crud():
    with patch('app.db.crud.SupabaseDBHandler') as MockHandler:
        mock_handler = MockHandler.return_value
        mock_handler.client = AsyncMock(return_value=MagicMock(table=MagicMock()))
        yield DummyCRUD('table')

@pytest.mark.asyncio
async def test_create(crud):
    crud.db_handler.client = AsyncMock(return_value=MagicMock(table=MagicMock(insert=MagicMock(return_value=MagicMock(execute=MagicMock(return_value=MagicMock(data='ok')))))))
    result = await crud.create({'a': 1})
    assert result == 'ok'

@pytest.mark.asyncio
async def test_read(crud):
    mock_table = MagicMock()
    mock_table.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[{'a': 1}])
    crud.db_handler.client = AsyncMock(return_value=MagicMock(table=MagicMock(return_value=mock_table)))
    result = await crud.read({'a': 1})
    assert isinstance(result, list)

@pytest.mark.asyncio
async def test_update(crud):
    mock_table = MagicMock()
    mock_table.update.return_value.eq.return_value.execute.return_value = MagicMock(data='updated')
    crud.db_handler.client = AsyncMock(return_value=MagicMock(table=MagicMock(return_value=mock_table)))
    result = await crud.update({'a': 1}, {'b': 2})
    assert result == 'updated'

@pytest.mark.asyncio
async def test_delete(crud):
    mock_table = MagicMock()
    mock_table.delete.return_value.eq.return_value.execute.return_value = MagicMock(data='deleted')
    crud.db_handler.client = AsyncMock(return_value=MagicMock(table=MagicMock(return_value=mock_table)))
    result = await crud.delete({'a': 1})
    assert result == 'deleted'

@pytest.mark.asyncio
async def test_unique_logic(crud):
    result = await crud.unique_logic()
    assert result == 'unique'
