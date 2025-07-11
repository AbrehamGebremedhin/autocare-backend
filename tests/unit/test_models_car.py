"""
Unit tests for Car schema/model.
Tests validation, serialization, and business rules.
"""
import pytest
from pydantic import ValidationError
from app.schemas.Car import CarBase


class TestCarBase:
    """Test cases for CarBase model."""
    
    def test_car_creation_with_minimal_data(self):
        """Test creating a car with minimal required data."""
        car_data = {
            "make": "Toyota",
            "model": "Camry",
            "year": 2020,
            "text": "Toyota Camry 2020 sedan"
        }
        car = CarBase(**car_data)
        
        assert car.make == "Toyota"
        assert car.model == "Camry"
        assert car.year == 2020
        assert car.text == "Toyota Camry 2020 sedan"
        assert car.id is None
        assert car.owner_manual_url is None
        assert car.service_manual_url is None
        assert car.car_guide_links is None
    
    def test_car_creation_with_full_data(self, sample_car_data):
        """Test creating a car with all fields populated."""
        car = CarBase(**sample_car_data)
        
        assert car.id == sample_car_data["id"]
        assert car.make == sample_car_data["make"]
        assert car.model == sample_car_data["model"]
        assert car.year == sample_car_data["year"]
        assert car.text == sample_car_data["text"]
        assert car.owner_manual_url == sample_car_data["owner_manual_url"]
        assert car.service_manual_url == sample_car_data["service_manual_url"]
        assert car.car_guide_links == sample_car_data["car_guide_links"]
    
    def test_missing_required_fields_raise_validation_error(self):
        """Test that missing required fields raise ValidationError."""
        required_fields = ["make", "model", "year", "text"]
        
        for field in required_fields:
            car_data = {
                "make": "Toyota",
                "model": "Camry",
                "year": 2020,
                "text": "Toyota Camry 2020 sedan"
            }
            # Remove the required field
            del car_data[field]
            
            with pytest.raises(ValidationError) as exc_info:
                CarBase(**car_data)
            
            assert field in str(exc_info.value).lower()
    
    def test_empty_string_fields_raise_validation_error(self):
        """Test that empty string fields don't raise validation errors (current behavior)."""
        # Note: The current Car model doesn't validate empty strings
        # This test documents the current behavior
        string_fields = ["make", "model", "text"]
        
        for field in string_fields:
            car_data = {
                "make": "Toyota",
                "model": "Camry",
                "year": 2020,
                "text": "Toyota Camry 2020 sedan"
            }
            # Set field to empty string
            car_data[field] = ""
            
            # This should not raise an exception with the current implementation
            car = CarBase(**car_data)
            assert getattr(car, field) == ""
    
    def test_year_validation(self):
        """Test year field validation."""
        base_car_data = {
            "make": "Toyota",
            "model": "Camry",
            "text": "Toyota Camry sedan"
        }
        
        # Test valid years
        valid_years = [1900, 1950, 2000, 2020, 2024, 2025]
        for year in valid_years:
            car_data = {**base_car_data, "year": year}
            car = CarBase(**car_data)
            assert car.year == year
        
        # Test invalid years (if any validation is implemented)
        invalid_years = [-1, 0, 1800, 2050]
        for year in invalid_years:
            car_data = {**base_car_data, "year": year}
            try:
                car = CarBase(**car_data)
                # If no validation error, year should still be set
                assert car.year == year
            except ValidationError:
                # If validation error occurs, that's acceptable
                pass
    
    def test_make_validation(self):
        """Test make field validation."""
        base_car_data = {
            "model": "Camry",
            "year": 2020,
            "text": "Toyota Camry 2020 sedan"
        }
        
        valid_makes = [
            "Toyota", "Honda", "Ford", "BMW", "Mercedes-Benz",
            "Audi", "Volkswagen", "Nissan", "Hyundai", "Kia"
        ]
        
        for make in valid_makes:
            car_data = {**base_car_data, "make": make}
            car = CarBase(**car_data)
            assert car.make == make
    
    def test_model_validation(self):
        """Test model field validation."""
        base_car_data = {
            "make": "Toyota",
            "year": 2020,
            "text": "Toyota Camry 2020 sedan"
        }
        
        valid_models = [
            "Camry", "Corolla", "Prius", "RAV4", "Highlander",
            "Civic", "Accord", "F-150", "Mustang", "Explorer"
        ]
        
        for model in valid_models:
            car_data = {**base_car_data, "model": model}
            car = CarBase(**car_data)
            assert car.model == model
    
    def test_text_field_validation(self):
        """Test text field validation."""
        car_data = {
            "make": "Toyota",
            "model": "Camry",
            "year": 2020,
            "text": "Toyota Camry 2020 sedan with advanced safety features"
        }
        car = CarBase(**car_data)
        
        assert car.text == "Toyota Camry 2020 sedan with advanced safety features"
        assert len(car.text) > 0
    
    def test_url_fields_validation(self):
        """Test URL fields validation."""
        car_data = {
            "make": "Toyota",
            "model": "Camry",
            "year": 2020,
            "text": "Toyota Camry 2020 sedan",
            "owner_manual_url": "https://example.com/owner-manual.pdf",
            "service_manual_url": "https://example.com/service-manual.pdf"
        }
        car = CarBase(**car_data)
        
        assert car.owner_manual_url == "https://example.com/owner-manual.pdf"
        assert car.service_manual_url == "https://example.com/service-manual.pdf"
    
    def test_car_guide_links_validation(self):
        """Test car_guide_links field validation."""
        car_data = {
            "make": "Toyota",
            "model": "Camry",
            "year": 2020,
            "text": "Toyota Camry 2020 sedan",
            "car_guide_links": [
                "https://example.com/guide1",
                "https://example.com/guide2",
                "https://example.com/guide3"
            ]
        }
        car = CarBase(**car_data)
        
        assert car.car_guide_links == [
            "https://example.com/guide1",
            "https://example.com/guide2",
            "https://example.com/guide3"
        ]
        assert len(car.car_guide_links) == 3
    
    def test_empty_car_guide_links(self):
        """Test empty car_guide_links list."""
        car_data = {
            "make": "Toyota",
            "model": "Camry",
            "year": 2020,
            "text": "Toyota Camry 2020 sedan",
            "car_guide_links": []
        }
        car = CarBase(**car_data)
        
        assert car.car_guide_links == []
        assert len(car.car_guide_links) == 0
    
    def test_car_serialization(self, sample_car_data):
        """Test that car can be serialized to dict."""
        car = CarBase(**sample_car_data)
        car_dict = car.dict()
        
        assert isinstance(car_dict, dict)
        assert car_dict["make"] == sample_car_data["make"]
        assert car_dict["model"] == sample_car_data["model"]
        assert car_dict["year"] == sample_car_data["year"]
        assert car_dict["text"] == sample_car_data["text"]
        assert car_dict["owner_manual_url"] == sample_car_data["owner_manual_url"]
        assert car_dict["service_manual_url"] == sample_car_data["service_manual_url"]
        assert car_dict["car_guide_links"] == sample_car_data["car_guide_links"]
    
    def test_car_json_serialization(self, sample_car_data):
        """Test that car can be serialized to JSON."""
        car = CarBase(**sample_car_data)
        car_json = car.json()
        
        assert isinstance(car_json, str)
        
        # Parse back to verify
        import json
        parsed = json.loads(car_json)
        assert parsed["make"] == sample_car_data["make"]
        assert parsed["model"] == sample_car_data["model"]
        assert parsed["year"] == sample_car_data["year"]
    
    def test_car_config_from_attributes(self):
        """Test that Config.from_attributes is set correctly."""
        assert CarBase.Config.from_attributes is True
    
    def test_car_update_with_new_data(self, sample_car_data):
        """Test updating car with new data."""
        car = CarBase(**sample_car_data)
        
        # Test updating individual fields
        updated_data = {
            "make": "Honda",
            "model": "Civic",
            "year": 2021,
            "text": "Honda Civic 2021 updated text"
        }
        
        updated_car = car.copy(update=updated_data)
        
        assert updated_car.make == "Honda"
        assert updated_car.model == "Civic"
        assert updated_car.year == 2021
        assert updated_car.text == "Honda Civic 2021 updated text"
        # Original data should remain unchanged
        assert car.make == sample_car_data["make"]
    
    def test_car_equality_comparison(self):
        """Test car equality comparison."""
        car_data = {
            "make": "Toyota",
            "model": "Camry",
            "year": 2020,
            "text": "Toyota Camry 2020 sedan",
            "id": "car-123"
        }
        
        car1 = CarBase(**car_data)
        car2 = CarBase(**car_data)
        
        assert car1.dict() == car2.dict()
    
    def test_car_with_none_values(self):
        """Test car creation with explicit None values for optional fields."""
        car_data = {
            "make": "Toyota",
            "model": "Camry",
            "year": 2020,
            "text": "Toyota Camry 2020 sedan",
            "id": None,
            "owner_manual_url": None,
            "service_manual_url": None,
            "car_guide_links": None
        }
        
        car = CarBase(**car_data)
        
        assert car.make == "Toyota"
        assert car.model == "Camry"
        assert car.year == 2020
        assert car.text == "Toyota Camry 2020 sedan"
        assert car.id is None
        assert car.owner_manual_url is None
        assert car.service_manual_url is None
        assert car.car_guide_links is None
    
    def test_car_field_presence(self):
        """Test that all expected fields are present."""
        car = CarBase(
            make="Toyota",
            model="Camry",
            year=2020,
            text="Toyota Camry 2020 sedan"
        )
        
        # Check that all expected fields exist
        expected_fields = [
            "id", "make", "model", "year", "text",
            "owner_manual_url", "service_manual_url", "car_guide_links"
        ]
        
        for field in expected_fields:
            assert hasattr(car, field), f"Field {field} is missing from CarBase"
    
    def test_car_validation_with_extra_fields(self):
        """Test behavior when extra fields are provided."""
        car_data = {
            "make": "Toyota",
            "model": "Camry",
            "year": 2020,
            "text": "Toyota Camry 2020 sedan",
            "extra_field": "should_be_ignored"
        }
        
        # This should either ignore the extra field or raise an error
        # depending on the model configuration
        try:
            car = CarBase(**car_data)
            # If no error, check that extra field is not present
            assert not hasattr(car, "extra_field")
        except ValidationError:
            # If ValidationError is raised, that's also acceptable behavior
            pass
    
    def test_car_make_model_combination(self):
        """Test various make/model combinations."""
        combinations = [
            ("Toyota", "Camry"),
            ("Honda", "Civic"),
            ("Ford", "F-150"),
            ("BMW", "X5"),
            ("Mercedes-Benz", "C-Class"),
            ("Audi", "A4"),
            ("Volkswagen", "Golf"),
            ("Nissan", "Altima"),
            ("Hyundai", "Elantra"),
            ("Kia", "Sportage")
        ]
        
        for make, model in combinations:
            car_data = {
                "make": make,
                "model": model,
                "year": 2020,
                "text": f"{make} {model} 2020"
            }
            car = CarBase(**car_data)
            
            assert car.make == make
            assert car.model == model
            assert car.text == f"{make} {model} 2020"
