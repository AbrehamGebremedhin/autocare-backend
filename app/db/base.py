from supabase import create_client, Client
from app.core.config import get_settings

class SupabaseDBHandler:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SupabaseDBHandler, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        settings = get_settings()
        SUPABASE_URL = settings.SUPABASE_URL
        SUPABASE_KEY = settings.SUPABASE_KEY
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase URL and Key must be set in environment variables")
        self._client = create_client(SUPABASE_URL, SUPABASE_KEY)

    @property
    def client(self) -> Client:
        if self._client is None:
            raise ValueError("Supabase client is not initialized")
        return self._client
