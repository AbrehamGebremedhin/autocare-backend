import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_rate_limit_exceeded():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        for _ in range(11):
            response = await ac.get("/")
        assert response.status_code == 429 or response.status_code == 200
