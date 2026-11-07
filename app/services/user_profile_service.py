from app.services.base_service import BaseService
from app.CRUD.car_crud import CarCRUD, log_id
from app.db import pool
from app.db.crud import BaseCRUD
from app.db.pool import DatabaseUnavailable
from app.utils.exceptions import DatabaseException
from app.utils.logger import get_logger_instance
from typing import List, Dict, Any, Optional


class UserProfileService(BaseService):
    """Service for managing user profiles and car associations"""
    
    def __init__(self, car_crud: Optional[CarCRUD] = None):
        super().__init__()
        self.car_crud = car_crud or CarCRUD()
        self.profiles = BaseCRUD("user_profiles")
        self.logger = get_logger_instance("user_profile")
    
    async def perform_action(self, action: str, *args, **kwargs):
        """
        Perform action for BaseService compatibility
        """
        if action == "add_car":
            return await self.add_car_to_user(*args, **kwargs)
        elif action == "remove_car":
            return await self.remove_car_from_user(*args, **kwargs)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
        
    async def add_car_to_user(
        self, 
        user_id: str, 
        make: str, 
        model: str, 
        year: int,
    ) -> Dict[str, Any]:
        """
        Add a car to a user's profile. Creates car if it doesn't exist.
        
        Args:
            user_id: the account's users.id
            make: Car make
            model: Car model
            year: Car year
            
        Returns:
            Dict with success status and car info
        """
        try:
            # Create or get existing car
            car_data = {
                'make': make,
                'model': model,
                'year': year
            }
            
            cars = await self.car_crud.create_or_get_car(car_data)
            
            if not cars or not isinstance(cars, list) or not cars[0]:
                return {"success": False, "error": "Failed to create/get car", "car_id": None}
            
            car = cars[0]
            car_id = car['id']
            
            await self.logger.info(f"Car {log_id(car_id)} ready for user {user_id}")
            
            await self._get_or_create_user_profile(user_id)
            
            # One atomic statement: concurrent adds cannot overwrite each other, and a duplicate matches no row.
            status = await pool.execute(
                "UPDATE user_profiles SET cars = array_append(cars, $2::text), updated_at = now()"
                " WHERE user_id = $1 AND NOT ($2::text = ANY(cars))",
                user_id, car_id,
            )
            if status == "UPDATE 1":
                await self.logger.info(f"Added car {log_id(car_id)} to user {user_id}'s profile")
            else:
                await self.logger.info(f"Car {log_id(car_id)} already in user {user_id}'s profile")
            
            return {
                "success": True, 
                "car_id": car_id,
                "car_info": car,
                "message": f"Added {make} {model} {year} to your garage"
            }
            
        except ValueError:
            return {"success": False, "error": "Make, model, and year are required.", "car_id": None}
        except DatabaseUnavailable:
            raise  # 503, not "could not add"
        except Exception as e:
            await self.logger.error(f"Error adding car to user {user_id}: {type(e).__name__}")
            raise DatabaseException("Could not add car") from e

    async def get_user_cars(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all cars associated with a user"""
        try:
            profile = await self._get_user_profile(user_id)
            if not profile:
                return []
            
            car_ids = profile.get('cars', [])
            if not isinstance(car_ids, list):
                return []
            
            # Get full car details
            cars = []
            for car_id in car_ids:
                car = await self.car_crud.get_car_by_id(car_id)
                if car:
                    cars.append(car)
            
            await self.logger.info(f"Retrieved {len(cars)} cars for user {user_id}")
            return cars

        except DatabaseUnavailable:
            raise  # 503: an outage must not read as "no cars" (or "car not found" when opening a session)
        except Exception as e:
            await self.logger.error(f"Error getting cars for user {user_id}: {type(e).__name__}")
            raise DatabaseException("Could not load cars") from e

    async def remove_car_from_user(self, user_id: str, car_id: str) -> Dict[str, Any]:
        """Remove a car from user's profile"""
        try:
            status = await pool.execute(
                "UPDATE user_profiles SET cars = array_remove(cars, $2::text), updated_at = now()"
                " WHERE user_id = $1 AND $2::text = ANY(cars)",
                user_id, car_id,
            )
            if status == "UPDATE 1":
                await self.logger.info(f"Removed car {log_id(car_id)} from user {user_id}'s profile")
                return {"success": True, "message": "Car removed from your garage"}
            else:
                return {"success": False, "error": "Car not found in your garage"}

        except DatabaseUnavailable:
            raise  # 503, not "car not found"
        except Exception as e:
            await self.logger.error(f"Error removing car from user {user_id}: {type(e).__name__}")
            raise DatabaseException("Could not remove car") from e

    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from database. Database errors propagate to the caller."""
        rows = await self.profiles.read({'user_id': user_id})
        return rows[0] if rows else None

    async def _get_or_create_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get or create user profile (cars and preferences start empty)"""
        status = await pool.execute(
            "INSERT INTO user_profiles (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING", user_id
        )
        if status == "INSERT 0 1":
            await self.logger.info(f"Created new profile for user {user_id}")
        return await self._get_user_profile(user_id) or {'user_id': user_id, 'cars': [], 'preferences': {}}
