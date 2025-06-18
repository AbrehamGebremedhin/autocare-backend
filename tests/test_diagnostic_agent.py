import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.diagnostic_agent import DiagnosisAgent
from app.utils.diagnosis_tree import DiagnosisTreeNode
import asyncio

@pytest.fixture
def diagnosis_tree():
    root = DiagnosisTreeNode('Engine', 0.9)
    child = DiagnosisTreeNode('Spark Plug', 0.7)
    root.add_child(child)
    return root

@pytest.fixture
def agent(diagnosis_tree):
    with patch('app.agents.diagnostic_agent.Logger'), \
         patch('app.agents.diagnostic_agent.LLMService') as MockLLM, \
         patch('app.agents.diagnostic_agent.CarCRUD') as MockCarCRUD, \
         patch('app.agents.diagnostic_agent.EmbeddingService') as MockEmbed, \
         patch('app.agents.diagnostic_agent.ScraperService') as MockScraper, \
         patch('app.agents.diagnostic_agent.manager'):
        mock_llm = MockLLM.return_value
        mock_llm.__or__ = lambda self, other: self
        mock_llm.invoke = AsyncMock(return_value='{"diagnosis_summary": "Test diagnosis", "supporting_evidence": [], "recommendations": []}')
        mock_llm.ainvoke = AsyncMock(return_value='{"diagnosis_summary": "Test diagnosis", "supporting_evidence": [], "recommendations": []}')
        mock_carcrud = MockCarCRUD.return_value
        mock_carcrud.get_car_by_id = AsyncMock(return_value={"vector": "manual", "car_guide_links": []})
        mock_embed = MockEmbed.return_value
        mock_embed.embed_text = AsyncMock(return_value=[1,2,3])
        mock_embed.embed_texts = AsyncMock(return_value=[[1,2,3]])
        mock_scraper = MockScraper.return_value
        mock_scraper.perform_action = AsyncMock(return_value=[{"text": "online context"}])
        yield DiagnosisAgent('car123', diagnosis_tree)

def test_summarize_tree(agent, diagnosis_tree):
    summary = agent.summarize_tree()
    assert 'Engine' in summary
    assert 'Spark Plug' in summary

@pytest.mark.asyncio
async def test_diagnose_success(agent):
    result = await agent.diagnose('My car is making noise')
    assert result['success']
    assert 'diagnosis' in result

@pytest.mark.asyncio
async def test_process_success(agent):
    result = await agent.process('My car is making noise')
    assert result['success']
    assert 'result' in result

@pytest.mark.asyncio
async def test_diagnose_error(agent):
    # Patch retrieve_context to raise
    with patch.object(agent, 'retrieve_context', side_effect=Exception('fail')):
        result = await agent.diagnose('fail')
        assert not result['success']
        assert result['error'] == 'fail'

@pytest.mark.asyncio
async def test_process_error(agent):
    with patch.object(agent, 'diagnose', side_effect=Exception('fail2')):
        result = await agent.process('fail2')
        assert not result['success']
        assert result['error'] == 'fail2'
