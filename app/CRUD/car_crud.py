from app.db.crud import BaseCRUD
from app.utils.logger import get_logger_instance, Logger
from typing import List, Dict, Optional, Any
import json

class CarCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('Car')
        self.logger = get_logger_instance("car_crud")

    def unique_logic(self, make, model, year):
        """Generate a unique car id using make, model, and year."""
        if not (make and model and year):
            raise ValueError("Make, model, and year are required for unique car id.")
        return f"{make.strip().lower().replace(' ', '-')}-{model.strip().lower().replace(' ', '-')}-{year}"

    def ensure_list(self, value):
        """Ensure the value is a list of proper CarGuideLink objects, parsing from JSON if needed."""
        if isinstance(value, list):
            # If it's already a list, ensure each item has the right structure
            result = []
            for item in value:
                if isinstance(item, dict) and 'link' in item:
                    result.append(item)
                elif isinstance(item, str):
                    # If it's a string, treat it as a link with no summary
                    result.append({'link': item, 'summary': None})
            return result
        if value is None:
            return []
        try:
            # Try to parse as JSON
            parsed = json.loads(value)
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, dict) and 'link' in item:
                        result.append(item)
                    elif isinstance(item, str):
                        result.append({'link': item, 'summary': None})
                return result
            return []
        except Exception:
            return []

    async def create_or_get_car(self, data):
        """
        Create a car record if it doesn't exist, or return existing car.
        Ensures cars are unique.
        """
        make = data.get('make')
        model = data.get('model')
        year = data.get('year')
        
        if not (make and model and year):
            raise ValueError("Make, model, and year are required.")
        
        # Generate unique car ID
        if not data.get('id'):
            data['id'] = self.unique_logic(make, model, year)
        car_id = data['id']
        
        # Check if car already exists
        existing = await self.read({'id': car_id})
        if existing and isinstance(existing, list) and existing:
            existing_car = existing[0]
            await self.logger.info(f"Car {car_id} already exists, returning existing car")
            return [existing_car]
        
        # Initialize required fields
        data['owner_manual_url'] = data.get('owner_manual_url') or ""
        data['service_manual_url'] = data.get('service_manual_url') or ""
        data['car_guide_links'] = self.ensure_list(data.get('car_guide_links'))
        
        # Create the car record
        return await super().create(data)

    async def read(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Override read to properly parse car_guide_links JSON field"""
        result = await super().read(filters)
        
        # Process each car record to parse car_guide_links
        if result and isinstance(result, list):
            for car in result:
                if isinstance(car, dict) and 'car_guide_links' in car:
                    car['car_guide_links'] = self.ensure_list(car['car_guide_links'])
        
        return result

    async def create(self, data):
        """Create a car record - now redirects to create_or_get_car for uniqueness"""
        return await self.create_or_get_car(data)

    async def get_car_by_make_model_year(self, make: str, model: str, year: int):
        """Retrieve a car by make, model, and year."""
        car_id = self.unique_logic(make, model, year)
        cars = await self.read({'id': car_id})
        if cars and isinstance(cars, list) and cars:
            return cars[0]
        return None

    async def get_car_by_id(self, car_id: str):
        """Retrieve a car by its unique id."""
        # Normalize car_id to handle different formats
        normalized_id = self._normalize_car_id(car_id)
        
        # First try with the original car_id
        cars = await self.read({'id': car_id})
        
        # If not found, try with normalized ID
        if not cars or not isinstance(cars, list) or not cars:
            if normalized_id != car_id:
                # Try again with normalized ID
                cars = await self.read({'id': normalized_id})
                
                # Log attempt with normalized ID
                if cars and isinstance(cars, list) and cars:
                    await self.logger.info(f"Found car with normalized ID: {normalized_id} (original: {car_id})")
                else:
                    await self.logger.warning(f"Car not found with original ID '{car_id}' or normalized ID '{normalized_id}'")
        
        if cars and isinstance(cars, list) and cars:
            return cars[0]
            
        return None
        
    def _normalize_car_id(self, car_id: str) -> str:
        """
        Normalize car ID to handle different formats and common variations.
        For example: "echo-toyota-2001" -> "toyota-echo-2001"
        """
        if not car_id:
            return car_id
            
        # Try to parse parts from the car_id
        parts = car_id.lower().strip().split('-')
        if len(parts) >= 3:
            # Check for common pattern where make and model are reversed
            # Common car makes that we can detect
            common_makes = ['toyota', 'honda', 'ford', 'chevrolet', 'bmw', 'audi', 'mercedes', 'nissan', 
                           'mazda', 'subaru', 'hyundai', 'kia', 'lexus', 'acura', 'volkswagen', 'vw']
            
            # If the first part isn't a known make but the second is, swap them
            if parts[0] not in common_makes and parts[1] in common_makes:
                model, make = parts[0], parts[1]
                remaining = '-'.join(parts[2:])
                return f"{make}-{model}-{remaining}"
        
        # If we couldn't normalize it, return the original
        return car_id

    async def batch_get_cars_by_ids(self, car_ids: List[str]) -> Dict[str, dict]:
        """
        Retrieve multiple cars by their IDs in a single operation.
        Returns a dictionary mapping car_id to car data.
        """
        results = {}
        if car_ids:
            batch_cars = await self.read({'id': {'in': car_ids}}) if len(car_ids) > 1 else await self.read({'id': car_ids[0]})
            for car in batch_cars or []:
                car_id = car.get('id')
                if car_id:
                    results[car_id] = car
        return results
