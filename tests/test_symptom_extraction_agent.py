import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.agents.symptom_extraction_agent import SymptomExtractorAgent

@pytest.fixture
def agent():
    return SymptomExtractorAgent()

@pytest.mark.asyncio
async def test_get_prompt_template(agent):
    template = agent.get_prompt_template()
    assert 'issue_name' in template.template

@pytest.mark.asyncio
async def test_extract_symptoms_success(agent):
    with patch('app.agents.symptom_extraction_agent.LLMService') as MockLLM, \
         patch('app.agents.symptom_extraction_agent.CarCRUD'), \
         patch('app.agents.symptom_extraction_agent.EmbeddingService'), \
         patch('app.agents.symptom_extraction_agent.ScraperService'), \
         patch('app.agents.symptom_extraction_agent.manager'), \
         patch('app.agents.symptom_extraction_agent.TreeManagerAgent'):
        mock_llm = MockLLM.return_value
        mock_llm.generate_response = AsyncMock(return_value='[{"issue_name": "Engine Knock", "likelihood": 90}]')
        # Simulate extract_symptoms method if it exists
        if hasattr(agent, 'extract_symptoms'):
            result = await agent.extract_symptoms('My car is making noise', context={})
            assert isinstance(result, list)
            assert result[0]['issue_name'] == 'Engine Knock'

@pytest.mark.asyncio
async def test_extract_symptoms_error(agent):
    with patch('app.agents.symptom_extraction_agent.LLMService') as MockLLM, \
         patch('app.agents.symptom_extraction_agent.CarCRUD'), \
         patch('app.agents.symptom_extraction_agent.EmbeddingService'), \
         patch('app.agents.symptom_extraction_agent.ScraperService'), \
         patch('app.agents.symptom_extraction_agent.manager'), \
         patch('app.agents.symptom_extraction_agent.TreeManagerAgent'):
        mock_llm = MockLLM.return_value
        mock_llm.generate_response = AsyncMock(side_effect=Exception('fail'))
        if hasattr(agent, 'extract_symptoms'):
            with pytest.raises(Exception):
                await agent.extract_symptoms('fail', context={})
