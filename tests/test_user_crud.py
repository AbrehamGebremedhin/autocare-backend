import pytest
from unittest.mock import patch
from app.CRUD.user_crud import UserCRUD

@pytest.fixture
def crud():
    with patch('app.CRUD.user_crud.BaseCRUD.__init__', return_value=None):
        yield UserCRUD()

def test_constructor(crud):
    assert crud is not None

@pytest.mark.asyncio
async def test_unique_logic(crud):
    result = await crud.unique_logic()
    assert result is None
