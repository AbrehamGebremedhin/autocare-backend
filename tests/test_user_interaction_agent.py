import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.user_interaction_agent import UserInteractionAgent

@pytest.fixture
def agent():
    with patch('app.agents.user_interaction_agent.LLMService') as MockLLM, \
         patch('app.agents.user_interaction_agent.Logger'), \
         patch('app.agents.user_interaction_agent.manager'):
        mock_llm = MockLLM.return_value
        mock_llm.generate_response = AsyncMock(return_value='User message generated')
        yield UserInteractionAgent()

@pytest.mark.asyncio
async def test_generate_user_message_success(agent):
    result = await agent.generate_user_message('My car is making noise', {'diagnosis': 'Test'})
    assert result['success']
    assert 'user_message' in result

@pytest.mark.asyncio
async def test_generate_user_message_error(agent):
    with patch.object(agent.llm_service, 'generate_response', side_effect=Exception('fail')):
        result = await agent.generate_user_message('fail', {'diagnosis': 'fail'})
        assert not result['success']
        assert 'user_message' in result

@pytest.mark.asyncio
async def test_process_success(agent):
    with patch.object(agent, 'generate_user_message', AsyncMock(return_value={'user_message': 'msg', 'success': True})):
        result = await agent.process('msg', {'diagnosis': 'Test'})
        assert result['success']
        assert 'user_message' in result

@pytest.mark.asyncio
async def test_process_error(agent):
    with patch.object(agent, 'generate_user_message', side_effect=Exception('fail2')):
        result = await agent.process('fail2', {'diagnosis': 'fail2'})
        assert not result['success']
        assert 'user_message' in result
