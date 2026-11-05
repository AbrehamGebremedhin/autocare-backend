import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture
async def ac():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_health_endpoint(ac, mocker):
    mocker.patch("main.llm.ping", return_value=True)
    mocker.patch("main.check_supabase", return_value=True)
    response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "supabase": True, "deepseek": True}


async def test_unknown_route_returns_404(ac):
    response = await ac.get("/nonexistent")
    assert response.status_code == 404
