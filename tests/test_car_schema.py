import pytest
from app.schemas.Car import CarBase

# Add your tests for Car schema here
def test_placeholder():
    assert True

def test_carbase_fields():
    car = CarBase(id='1', make='Toyota', model='Camry', year=2020, vector=[0.1, 0.2], owner_manual_url='url', service_manual_url='url2', car_guide_links=['a', 'b'])
    assert car.make == 'Toyota'
    assert isinstance(car.vector, list)
    assert car.car_guide_links == ['a', 'b']
