import pytest
from pydantic import ValidationError

from app.schemas.Car import CarBase, CarGuideLink


class TestCarBase:
    def test_minimal_car(self):
        car = CarBase(make="Toyota", model="Camry", year=2020)

        assert car.id is None
        assert car.make == "Toyota"
        assert car.model == "Camry"
        assert car.year == 2020
        assert car.owner_manual_url is None
        assert car.service_manual_url is None
        assert car.car_guide_links is None

    def test_full_car(self, sample_car_data):
        car = CarBase(**sample_car_data)

        assert car.id == "car-123"
        assert car.owner_manual_url == "https://example.com/manual.pdf"
        assert car.service_manual_url == "https://example.com/service.pdf"
        assert [g.link for g in car.car_guide_links] == [
            "https://example.com/guide1",
            "https://example.com/guide2",
        ]
        assert car.car_guide_links[0].summary == "Guide one"
        assert car.car_guide_links[1].summary is None

    @pytest.mark.parametrize("missing", ["make", "model", "year"])
    def test_required_fields(self, missing):
        data = {"make": "Toyota", "model": "Camry", "year": 2020}
        del data[missing]

        with pytest.raises(ValidationError) as exc:
            CarBase(**data)
        assert missing in str(exc.value)

    def test_year_must_be_integer(self):
        with pytest.raises(ValidationError):
            CarBase(make="Toyota", model="Camry", year="not-a-year")

    def test_guide_link_requires_link(self):
        with pytest.raises(ValidationError):
            CarGuideLink(summary="no link")

    def test_from_attributes(self):
        class Row:
            id = "car-1"
            make = "Honda"
            model = "Civic"
            year = 2019
            owner_manual_url = None
            service_manual_url = None
            car_guide_links = None

        car = CarBase.model_validate(Row())

        assert (car.id, car.make, car.model, car.year) == ("car-1", "Honda", "Civic", 2019)

    def test_round_trip_serialization(self, sample_car_data):
        car = CarBase(**sample_car_data)

        dumped = car.model_dump()

        assert dumped["car_guide_links"][0] == {
            "link": "https://example.com/guide1",
            "summary": "Guide one",
        }
        assert CarBase(**dumped) == car
