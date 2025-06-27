from abc import ABC, abstractmethod
from app.db.base import SupabaseDBHandler
from app.core.interfaces import IDBHandler
from typing import Any, Dict, Optional, List

class BaseCRUD(ABC):
    """
    Abstract CRUD base class for database operations.
    """
    def __init__(self, table_name: str, db_handler: IDBHandler = None):
        self.table_name = table_name
        self.db_handler = db_handler or SupabaseDBHandler()

    async def get_db(self):
        return await self.db_handler.client

    async def create(self, data: Dict[str, Any]) -> Any:
        db = await self.get_db()
        response = db.table(self.table_name).insert(data).execute()
        return getattr(response, 'data', response)

    async def read(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        db = await self.get_db()
        query = db.table(self.table_name).select('*')
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        response = query.execute()
        return getattr(response, 'data', response)

    async def update(self, filters: Dict[str, Any], update_data: Dict[str, Any]) -> Any:
        db = await self.get_db()
        query = db.table(self.table_name).update(update_data)
        for key, value in filters.items():
            query = query.eq(key, value)
        response = query.execute()
        return getattr(response, 'data', response)

    async def delete(self, filters: Dict[str, Any]) -> Any:
        db = await self.get_db()
        query = db.table(self.table_name).delete()
        for key, value in filters.items():
            query = query.eq(key, value)
        response = query.execute()
        return getattr(response, 'data', response)

    @abstractmethod
    async def unique_logic(self, *args, **kwargs):
        """
        Abstract method for table-specific logic to be implemented by subclasses.
        """
        pass
