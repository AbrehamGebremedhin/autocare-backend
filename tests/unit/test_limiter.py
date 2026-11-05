from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware

import main
from app.utils.limiter import limiter


def test_limiter_is_slowapi_limiter():
    assert isinstance(limiter, Limiter)


def test_app_uses_shared_limiter_and_middleware():
    assert main.app.state.limiter is limiter
    assert any(m.cls is SlowAPIMiddleware for m in main.app.user_middleware)
