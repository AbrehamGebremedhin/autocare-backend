import pytest
from app.schemas.Ground_Knowledge import GroundKnowledgeBase

# Add your tests for Ground_Knowledge schema here
def test_placeholder():
    assert True

def test_ground_knowledge_base_fields():
    gk = GroundKnowledgeBase(id='1', book_title='Book', content_chunk='chunk', vector=[0.1, 0.2], page_number=1, metadata={'a': 1})
    assert gk.book_title == 'Book'
    assert isinstance(gk.vector, list)
    assert gk.metadata['a'] == 1
