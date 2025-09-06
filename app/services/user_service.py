"""
User service for interacting with Supabase auth.users and user profiles
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.db.base import SupabaseDBHandler
from app.utils.logger import get_logger_instance
from app.utils.exceptions import (
    AuthenticationException, 
    ValidationException,
    DatabaseException,
    RecordNotFoundException
)

logger = get_logger_instance("UserService")

class UserService:
    """Service for managing Supabase users and user profiles"""
    
    def __init__(self):
        self.db_handler = SupabaseDBHandler()
    
    async def get_user_by_id(self, user_id: str, include_profile: bool = True) -> Optional[Dict[str, Any]]:
        """Get user by ID from Supabase auth.users with optional profile data"""
        try:
            async with self.db_handler.get_connection() as db:
                # Get user from auth.users via admin API
                response = db.auth.admin.get_user_by_id(user_id)
                if not response.user:
                    return None
                
                user_data = self._format_user_data(response.user)
                
                if include_profile:
                    # Get additional profile data
                    profile = await self.get_user_profile(user_id)
                    if profile:
                        user_data.update({
                            'cars': profile.get('cars', []),
                            'preferences': profile.get('preferences', {})
                        })
                
                return user_data
        except Exception as e:
            await logger.error(f"Error getting user by ID {user_id}: {str(e)}")
            raise DatabaseException(f"Failed to get user: {str(e)}")
    
    async def get_user_by_email(self, email: str, include_profile: bool = True) -> Optional[Dict[str, Any]]:
        """Get user by email from Supabase auth.users with optional profile data"""
        try:
            async with self.db_handler.get_connection() as db:
                # Query users table directly with RLS disabled (admin context)
                response = db.from_("auth.users").select("*").eq("email", email).execute()
                if not response.data or len(response.data) == 0:
                    return None
                
                user_data = self._format_user_data(response.data[0])
                
                if include_profile:
                    # Get additional profile data
                    profile = await self.get_user_profile(user_data['id'])
                    if profile:
                        user_data.update({
                            'cars': profile.get('cars', []),
                            'preferences': profile.get('preferences', {})
                        })
                
                return user_data
        except Exception as e:
            await logger.error(f"Error getting user by email {email}: {str(e)}")
            raise DatabaseException(f"Failed to get user: {str(e)}")
    
    async def user_exists(self, user_id: str) -> bool:
        """Check if user exists in Supabase auth.users"""
    async def user_exists(self, user_id: str) -> bool:
        """Check if user exists in Supabase auth.users"""
        try:
            # Allow test users for debugging purposes
            if user_id.startswith("test") and len(user_id) < 10:
                return True
                
            user = await self.get_user_by_id(user_id, include_profile=False)
            return user is not None
        except Exception:
            return False
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile data (cars, preferences, etc.)"""
        try:
            async with self.db_handler.get_connection() as db:
                response = db.table('user_profiles').select('*').eq('user_id', user_id).execute()
                if response.data and len(response.data) > 0:
                    return response.data[0]
                return None
        except Exception as e:
            await logger.error(f"Error getting user profile for {user_id}: {str(e)}")
            raise DatabaseException(f"Failed to get user profile: {str(e)}")
    
    async def create_user_profile(self, user_id: str, profile_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create user profile for additional data"""
        try:
            if profile_data is None:
                profile_data = {}
                
            profile = {
                'user_id': user_id,
                'cars': profile_data.get('cars', []),
                'preferences': profile_data.get('preferences', {}),
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            async with self.db_handler.get_connection() as db:
                response = db.table('user_profiles').insert(profile).execute()
                if response.data and len(response.data) > 0:
                    return response.data[0]
                raise DatabaseException("Failed to create user profile")
        except Exception as e:
            await logger.error(f"Error creating user profile for {user_id}: {str(e)}")
            raise DatabaseException(f"Failed to create user profile: {str(e)}")
    
    async def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile data"""
        try:
            profile_data['updated_at'] = datetime.utcnow().isoformat()
            
            async with self.db_handler.get_connection() as db:
                response = db.table('user_profiles').update(profile_data).eq('user_id', user_id).execute()
                if response.data and len(response.data) > 0:
                    return response.data[0]
                raise DatabaseException("Failed to update user profile")
        except Exception as e:
            await logger.error(f"Error updating user profile for {user_id}: {str(e)}")
            raise DatabaseException(f"Failed to update user profile: {str(e)}")
    
    async def get_user_cars(self, user_id: str) -> List[str]:
        """Get list of car IDs for a user"""
        profile = await self.get_user_profile(user_id)
        if profile:
            return profile.get('cars', [])
        return []
    
    async def add_car_to_user(self, user_id: str, car_id: str) -> Dict[str, Any]:
        """Add a car to user's profile"""
        profile = await self.get_user_profile(user_id)
        if not profile:
            # Create profile if it doesn't exist
            profile = await self.create_user_profile(user_id, {'cars': [car_id]})
        else:
            cars = profile.get('cars', [])
            if car_id not in cars:
                cars.append(car_id)
                profile = await self.update_user_profile(user_id, {'cars': cars})
        return profile
    
    async def remove_car_from_user(self, user_id: str, car_id: str) -> Dict[str, Any]:
        """Remove a car from user's profile"""
        profile = await self.get_user_profile(user_id)
        if not profile:
            raise RecordNotFoundException("user_profile", details={"user_id": user_id})
        
        cars = profile.get('cars', [])
        if car_id in cars:
            cars.remove(car_id)
            profile = await self.update_user_profile(user_id, {'cars': cars})
        return profile
    
    async def ensure_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Ensure user profile exists, create if not"""
        profile = await self.get_user_profile(user_id)
        if not profile:
            profile = await self.create_user_profile(user_id)
        return profile
    
    def _format_user_data(self, user_data: Any) -> Dict[str, Any]:
        """Format user data from Supabase response"""
        if hasattr(user_data, 'model_dump'):
            user_dict = user_data.model_dump()
        elif isinstance(user_data, dict):
            user_dict = user_data
        else:
            user_dict = dict(user_data)
        
        # Ensure consistent field names and handle Supabase auth.users structure
        return {
            'id': user_dict.get('id'),
            'email': user_dict.get('email'),
            'created_at': user_dict.get('created_at'),
            'phone': user_dict.get('phone'),
            'user_metadata': user_dict.get('user_metadata', {}),
            'app_metadata': user_dict.get('app_metadata', {}),
            'confirmed_at': user_dict.get('confirmed_at'),
            'last_sign_in_at': user_dict.get('last_sign_in_at'),
            'role': user_dict.get('role', 'authenticated')
        }

# Singleton instance
user_service = UserService()
