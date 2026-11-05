from app.services.base_service import BaseService
from app.CRUD.car_crud import CarCRUD
from app.db.base import SupabaseDBHandler
from app.utils.exceptions import DatabaseException
from app.utils.logger import get_logger_instance
from typing import List, Dict, Any, Optional
import asyncio


class UserProfileService(BaseService):
    """Service for managing user profiles and car associations"""
    
    def __init__(self, car_crud: Optional[CarCRUD] = None):
        super().__init__()
        self.car_crud = car_crud or CarCRUD()
        self.db_handler = SupabaseDBHandler()
        self.logger = get_logger_instance("user_profile")
    
    async def perform_action(self, action: str, *args, **kwargs):
        """
        Perform action for BaseService compatibility
        """
        if action == "add_car":
            return await self.add_car_to_user(*args, **kwargs)
        elif action == "get_cars":
            return await self.get_user_cars_with_details(*args, **kwargs)
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
            user_id: Supabase user ID
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
            
            await self.logger.info(f"Car {car_id} ready for user {user_id}")
            
            # Get or create user profile
            profile = await self._get_or_create_user_profile(user_id)
            
            # Add car to user's profile if not already there
            user_cars = profile.get('cars', [])
            if not isinstance(user_cars, list):
                user_cars = []
            
            if car_id not in user_cars:
                user_cars.append(car_id)
                
                # Update user profile
                await self._update_user_profile(user_id, {'cars': user_cars})
                
                await self.logger.info(f"Added car {car_id} to user {user_id}'s profile")
            else:
                await self.logger.info(f"Car {car_id} already in user {user_id}'s profile")
            
            return {
                "success": True, 
                "car_id": car_id,
                "car_info": car,
                "message": f"Added {make} {model} {year} to your garage"
            }
            
        except ValueError:
            return {"success": False, "error": "Make, model, and year are required.", "car_id": None}
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
            
        except Exception as e:
            await self.logger.error(f"Error getting cars for user {user_id}: {type(e).__name__}")
            return []

    async def remove_car_from_user(self, user_id: str, car_id: str) -> Dict[str, Any]:
        """Remove a car from user's profile"""
        try:
            profile = await self._get_user_profile(user_id)
            if not profile:
                return {"success": False, "error": "User profile not found"}
            
            user_cars = profile.get('cars', [])
            if not isinstance(user_cars, list):
                user_cars = []
            
            if car_id in user_cars:
                user_cars.remove(car_id)
                await self._update_user_profile(user_id, {'cars': user_cars})
                await self.logger.info(f"Removed car {car_id} from user {user_id}'s profile")
                return {"success": True, "message": "Car removed from your garage"}
            else:
                return {"success": False, "error": "Car not found in your garage"}
                
        except Exception as e:
            await self.logger.error(f"Error removing car from user {user_id}: {type(e).__name__}")
            raise DatabaseException("Could not remove car") from e

    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from database. Database errors propagate to the caller."""
        async with self.db_handler.get_connection() as db:
            result = db.table('user_profiles').select('*').eq('user_id', user_id).execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None

    async def _get_or_create_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get or create user profile"""
        profile = await self._get_user_profile(user_id)
        
        if not profile:
            # Create new profile
            profile_data = {
                'user_id': user_id,
                'cars': [],
                'preferences': {}
            }
            
            async with self.db_handler.get_connection() as db:
                result = db.table('user_profiles').insert(profile_data).execute()

                if result.data and len(result.data) > 0:
                    profile = result.data[0]
                    await self.logger.info(f"Created new profile for user {user_id}")
                else:
                    profile = profile_data

        return profile

    async def _update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile. Database errors propagate to the caller."""
        async with self.db_handler.get_connection() as db:
            result = db.table('user_profiles').update(updates).eq('user_id', user_id).execute()

            return result.data is not None
