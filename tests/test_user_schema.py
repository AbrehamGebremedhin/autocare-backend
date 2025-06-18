import pytest
from app.schemas.User import UserBase
from datetime import datetime

# Add your tests for User schema here
def test_placeholder():
    assert True

def test_userbase_fields():
    user = UserBase(id='1', email='a@b.com', created_at=datetime.now(), cars=['car1', 'car2'])
    assert user.email == 'a@b.com'
    assert user.cars == ['car1', 'car2']
