import pytest
from app.utils.diagnosis_tree_factory import get_diagnosis_tree
from app.utils.diagnosis_tree import DiagnosisTreeNode

# Add your tests for diagnosis_tree_factory here
def test_placeholder():
    assert True

def test_get_diagnosis_tree():
    tree = get_diagnosis_tree(issue_name='Test', likelyhood=0.5)
    assert isinstance(tree, DiagnosisTreeNode)
    assert tree.issue_name == 'Test'
    assert tree.likelyhood == 0.5
