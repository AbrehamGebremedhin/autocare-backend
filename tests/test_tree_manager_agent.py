import pytest
from unittest.mock import patch, MagicMock
from app.agents.tree_manager_agent import TreeManagerAgent
from app.utils.diagnosis_tree import DiagnosisTreeNode

@pytest.fixture
def tree():
    root = DiagnosisTreeNode('Engine', 0.9)
    child = DiagnosisTreeNode('Spark Plug', 0.7)
    root.add_child(child)
    return root

@pytest.fixture
def agent(tree):
    with patch('app.agents.tree_manager_agent.LLMService') as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.get_llm.return_value = MagicMock()
        mock_llm.generate_response = MagicMock(return_value='Engine')
        yield TreeManagerAgent(tree)

def test_get_tree_state(agent):
    state = agent.get_tree_state()
    assert 'Engine' in state
    assert 'Spark Plug' in state

def test_decide_parent_for_symptom(agent):
    with patch.object(agent.llm_service, 'generate_response', return_value='Engine'):
        node = agent.decide_parent_for_symptom('Noise')
        assert node.issue_name == 'Engine'

def test_add_symptom(agent):
    with patch.object(agent, 'decide_parent_for_symptom', return_value=agent.root):
        new_node = agent.add_symptom('Rattle', 0.5)
        assert new_node.issue_name == 'Rattle'
        assert new_node in agent.root.children

def test_prune_tree(agent):
    agent.add_symptom('LowProb', 0.1)
    agent.prune_tree(threshold=0.2)
    assert all(child.likelyhood >= 0.2 for child in agent.root.children)

def test_sort_tree(agent):
    agent.add_symptom('A', 0.2)
    agent.add_symptom('B', 0.8)
    agent.sort_tree()
    assert agent.root.children[0].likelyhood >= agent.root.children[1].likelyhood

# Add your tests for tree_manager_agent here
def test_placeholder():
    assert True
