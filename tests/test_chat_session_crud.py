import pytest
from unittest.mock import patch
from app.CRUD.chat_session_crud import ChatSessionCRUD

@pytest.fixture
def crud():
    with patch('app.CRUD.chat_session_crud.BaseCRUD.__init__', return_value=None):
        yield ChatSessionCRUD()

def test_constructor(crud):
    assert crud is not None

@pytest.mark.asyncio
async def test_unique_logic(crud):
    result = await crud.unique_logic()
    assert result is None
