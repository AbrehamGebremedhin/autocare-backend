from abc import ABC, abstractmethod
from app.db.base import SupabaseDBHandler
import asyncio
from typing import Any, Dict, Optional, List

class BaseCRUD(ABC):
    """
    Abstract CRUD base class for database operations.
    """
    def __init__(self, table_name: str, db_handler: Optional[SupabaseDBHandler] = None):
        self.table_name = table_name
        self.db_handler = db_handler or SupabaseDBHandler()

    async def get_db(self):
        return self.db_handler.client

    async def create(self, data: Dict[str, Any]) -> Any:
        async with self.db_handler.get_connection() as db:
            response = await asyncio.to_thread(db.table(self.table_name).insert(data).execute)
            return getattr(response, 'data', response)

    async def read(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        async with self.db_handler.get_connection() as db:
            query = db.table(self.table_name).select('*')
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)
            response = await asyncio.to_thread(query.execute)
            return getattr(response, 'data', response)

    async def update(self, filters: Dict[str, Any], update_data: Dict[str, Any]) -> Any:
        async with self.db_handler.get_connection() as db:
            query = db.table(self.table_name).update(update_data)
            for key, value in filters.items():
                query = query.eq(key, value)
            response = await asyncio.to_thread(query.execute)
            return getattr(response, 'data', response)

    async def delete(self, filters: Dict[str, Any]) -> Any:
        async with self.db_handler.get_connection() as db:
            query = db.table(self.table_name).delete()
            for key, value in filters.items():
                query = query.eq(key, value)
            response = await asyncio.to_thread(query.execute)
            return getattr(response, 'data', response)

    @abstractmethod
    async def unique_logic(self, *args, **kwargs):
        """
        Abstract method for table-specific logic to be implemented by subclasses.
        """
        pass
