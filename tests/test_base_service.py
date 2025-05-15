import pytest
from app.services.base_service import BaseService

class DummyService(BaseService):
    async def perform_action(self, *args, **kwargs):
        return 'performed'

import asyncio

def test_base_service_perform_action():
    service = DummyService()
    result = asyncio.run(service.perform_action())
    assert result == 'performed'
