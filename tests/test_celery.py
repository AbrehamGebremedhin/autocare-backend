import pytest
from app.core.celery import celery_app

# Add your tests for celery here
def test_placeholder():
    assert True

def test_celery_app_config():
    assert celery_app.main == 'autocare_tasks'
    assert celery_app.conf['task_serializer'] == 'json'
    assert celery_app.conf['result_serializer'] == 'json'
    assert 'json' in celery_app.conf['accept_content']
    assert celery_app.conf['timezone'] == 'UTC'
    assert celery_app.conf['enable_utc'] is True
