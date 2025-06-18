import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.db.bucket_operations import SupabaseBucketManager

@pytest.fixture
def manager():
    with patch('app.db.bucket_operations.SupabaseDBHandler.__new__', return_value=MagicMock()):
        yield SupabaseBucketManager()

@pytest.mark.asyncio
async def test_create_bucket(manager):
    manager.client = AsyncMock(return_value=MagicMock(storage=MagicMock(create_bucket=MagicMock(return_value={'ok': True}))))
    result = await manager.create_bucket('bucket')
    assert result is not None

@pytest.mark.asyncio
async def test_delete_bucket(manager):
    manager.client = AsyncMock(return_value=MagicMock(storage=MagicMock(delete_bucket=MagicMock(return_value={'ok': True}))))
    result = await manager.delete_bucket('bucket')
    assert result is not None

@pytest.mark.asyncio
async def test_list_buckets(manager):
    manager.client = AsyncMock(return_value=MagicMock(storage=MagicMock(list_buckets=MagicMock(return_value=['b']))))
    result = await manager.list_buckets()
    assert isinstance(result, list)

@pytest.mark.asyncio
async def test_upload_file(manager, tmp_path):
    file_path = tmp_path / 'file.txt'
    file_path.write_text('data')
    mock_storage = MagicMock()
    mock_storage.from_.return_value.upload.return_value = {'ok': True}
    manager.client = AsyncMock(return_value=MagicMock(storage=mock_storage))
    result = await manager.upload_file('bucket', str(file_path), 'dest.txt')
    assert result is not None

@pytest.mark.asyncio
async def test_download_file(manager):
    mock_storage = MagicMock()
    mock_storage.from_.return_value.download.return_value = b'data'
    manager.client = AsyncMock(return_value=MagicMock(storage=mock_storage))
    result = await manager.download_file('bucket', 'file.txt')
    assert result == b'data'

@pytest.mark.asyncio
async def test_delete_file(manager):
    mock_storage = MagicMock()
    mock_storage.from_.return_value.remove.return_value = {'ok': True}
    manager.client = AsyncMock(return_value=MagicMock(storage=mock_storage))
    result = await manager.delete_file('bucket', 'file.txt')
    assert result is not None
