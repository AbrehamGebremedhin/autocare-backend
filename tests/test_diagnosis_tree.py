import pytest
from app.utils.diagnosis_tree import DiagnosisTreeNode

# Add your tests for diagnosis_tree here
def test_placeholder():
    assert True

def test_tree_add_and_find():
    root = DiagnosisTreeNode('Engine', 0.9)
    child = DiagnosisTreeNode('Spark Plug', 0.7)
    root.add_child(child)
    assert root.find('Spark Plug') is child

def test_tree_remove_child():
    root = DiagnosisTreeNode('Engine', 0.9)
    child = DiagnosisTreeNode('Spark Plug', 0.7)
    root.add_child(child)
    root.remove_child(child)
    assert child not in root.children

def test_tree_traverse():
    root = DiagnosisTreeNode('Engine', 0.9)
    child = DiagnosisTreeNode('Plug', 0.5)
    root.add_child(child)
    nodes = list(root.traverse())
    assert root in nodes and child in nodes

def test_tree_prune_and_sort():
    root = DiagnosisTreeNode('Engine', 0.9)
    a = DiagnosisTreeNode('A', 0.1)
    b = DiagnosisTreeNode('B', 0.8)
    root.add_child(a)
    root.add_child(b)
    root.prune(0.2)
    assert a not in root.children
    root.sort_children_by_likelyhood()
    assert root.children[0].likelyhood >= root.children[-1].likelyhood
