import pytest
from unittest.mock import patch, MagicMock
from app.services.llm_service import LLMService

@pytest.fixture
def service():
    with patch('app.services.llm_service.Ollama') as MockOllama:
        MockOllama.return_value = MagicMock()
        yield LLMService()

def test_set_model(service):
    service.set_model('newmodel', version='v2', param=1)
    assert service.model_name == 'newmodel'
    assert service.version == 'v2'

def test_render_prompt(service):
    template = 'Hello $name'
    variables = {'name': 'World'}
    result = service.render_prompt(template, variables)
    assert result == 'Hello World'

def test_cache_key(service):
    key = service._cache_key('prompt', {'a': 1})
    assert isinstance(key, str)
