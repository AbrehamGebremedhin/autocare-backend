import pytest
from app.core.config import get_settings, Settings
import os

def test_get_settings_env(monkeypatch):
    monkeypatch.setenv('SUPABASE_URL', 'url')
    monkeypatch.setenv('SUPABASE_KEY', 'key')
    monkeypatch.setenv('SUPABASE_PASSWORD', 'pw')
    monkeypatch.setenv('GEMINI_KEY', 'gkey')
    monkeypatch.setenv('GEMINI_MODEL_1', 'gmodel1')
    monkeypatch.setenv('GEMINI_MODEL_2', 'gmodel2')
    monkeypatch.setenv('YOUTUBE_API_KEY', 'ytkey')
    settings = get_settings()
    assert settings.SUPABASE_URL == 'url'
    assert settings.SUPABASE_KEY == 'key'
    assert settings.SUPABASE_PASSWORD == 'pw'
    assert settings.GEMINI_KEY == 'gkey'
    assert settings.GEMINI_MODEL_1 == 'gmodel1'
    assert settings.GEMINI_MODEL_2 == 'gmodel2'
    assert settings.YOUTUBE_API_KEY == 'ytkey'

def test_settings_defaults():
    s = Settings(SUPABASE_URL='a', SUPABASE_KEY='b', SUPABASE_PASSWORD='c', GEMINI_KEY='d', GEMINI_MODEL_1='e', GEMINI_MODEL_2='f', YOUTUBE_API_KEY='g')
    assert s.APP_NAME == 'AutoCare API'
    assert s.DEBUG is False
    assert s.ENV in ['development', os.getenv('ENV', 'development')]
