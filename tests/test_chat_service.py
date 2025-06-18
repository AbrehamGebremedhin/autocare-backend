import pytest
from unittest.mock import patch, MagicMock
from app.services.chat_service import ChatService

@pytest.fixture
def service():
    with patch('app.services.chat_service.OrchestratorAgent') as MockOrch:
        MockOrch.return_value = MagicMock()
        yield ChatService()

def test_get_conversation_new(service):
    conv = service._get_conversation('user1')
    assert conv['user_id'] == 'user1'
    assert 'diagnosis_tree' in conv['context']

def test_get_conversation_existing(service):
    conv1 = service._get_conversation('user2')
    conv1['last_updated'] = conv1['created_at']
    conv2 = service._get_conversation('user2')
    assert conv1 is conv2
