import pytest
from app.schemas.Diagnosis_History import Diagnosis_History
from datetime import datetime

# Add your tests for Diagnosis_History schema here
def test_placeholder():
    assert True

def test_diagnosis_history_fields():
    dh = Diagnosis_History(id='1', user_id='u', session_data={'a': 1}, timestamp=datetime.now())
    assert dh.user_id == 'u'
    assert isinstance(dh.session_data, dict)
