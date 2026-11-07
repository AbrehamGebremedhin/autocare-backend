from slowapi import Limiter
from slowapi.util import get_remote_address

# Shared limiter instance, wired into the app in main.py (SlowAPIMiddleware + handler).
limiter = Limiter(key_func=get_remote_address)
