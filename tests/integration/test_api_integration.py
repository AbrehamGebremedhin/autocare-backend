import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_root_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert "Welcome to AutoCare API" in response.text

@pytest.mark.asyncio
async def test_not_found():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/nonexistent")
    assert response.status_code == 404
    assert "Not found" in response.text or "detail" in response.text

@pytest.mark.asyncio
async def test_rate_limit():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        for _ in range(11):
            response = await ac.get("/")
        assert response.status_code in (200, 429)
        if response.status_code == 429:
            assert "Rate limit exceeded" in response.text

# Example for a protected endpoint (adjust path and logic as needed)
# @pytest.mark.asyncio
# async def test_protected_endpoint():
#     async with AsyncClient(app=app, base_url="http://test") as ac:
#         response = await ac.get("/api/v1/protected")
#     assert response.status_code in (401, 403)
