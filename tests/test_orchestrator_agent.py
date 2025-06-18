import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.agents.orchestrator_agent import OrchestratorAgent

@pytest.fixture
def agent():
    with patch('app.agents.orchestrator_agent.UserInteractionAgent') as MockUser, \
         patch('app.agents.orchestrator_agent.SymptomExtractorAgent') as MockSymptom, \
         patch('app.agents.orchestrator_agent.DiagnosisAgent') as MockDiag, \
         patch('app.agents.orchestrator_agent.manager'):
        mock_user = MockUser.return_value
        mock_user.process = AsyncMock(return_value={'result': 'user msg', 'success': True})
        mock_symptom = MockSymptom.return_value
        mock_symptom.extract_symptoms = AsyncMock(return_value={'tree': None})
        mock_diag = MockDiag.return_value
        mock_diag.process = AsyncMock(return_value={'success': True, 'diagnosis': 'diag', 'need_symptom_extraction': False})
        yield OrchestratorAgent()

@pytest.mark.asyncio
async def test_route_request_initial_message(agent):
    with patch('app.agents.orchestrator_agent.OrchestratorAgent._handle_with_agent', AsyncMock(return_value={'tree': None})):
        result = await agent.route_request('msg', user_id='u', context={'is_initial_message': True, 'car_id': 'c', 'diagnosis_tree': None})
        assert 'success' in result

@pytest.mark.asyncio
async def test_route_request_diagnosis(agent):
    result = await agent.route_request('msg', user_id='u', context={'car_id': 'c', 'diagnosis_tree': None})
    assert 'success' in result

@pytest.mark.asyncio
async def test_route_request_no_car_id(agent):
    result = await agent.route_request('msg', user_id='u', context={})
    assert 'success' in result

@pytest.mark.asyncio
async def test_is_chat_request(agent):
    assert agent._is_chat_request('hello')
    assert not agent._is_chat_request('extract symptom')

# Add your tests for orchestrator_agent here
def test_placeholder():
    assert True
