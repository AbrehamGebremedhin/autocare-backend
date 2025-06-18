import pytest
from app.schemas.Chat_Session import ChatSession
from app.utils.diagnosis_tree import DiagnosisTreeNode
from datetime import datetime

# Add your tests for Chat_Session schema here
def test_placeholder():
    assert True

def test_chatsession_fields():
    session = ChatSession(id='1', user_id='u', messages=[], created_at=datetime.utcnow(), diagnosis_tree=None)
    assert session.user_id == 'u'

def test_serialize_deserialize_tree():
    root = DiagnosisTreeNode('Engine', 0.9)
    child = DiagnosisTreeNode('Spark Plug', 0.7)
    root.add_child(child)
    data = ChatSession.serialize_diagnosis_tree(root)
    new_tree = ChatSession.deserialize_diagnosis_tree(data)
    assert new_tree.issue_name == 'Engine'
    assert new_tree.children[0].issue_name == 'Spark Plug'
