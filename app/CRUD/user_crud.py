from app.db.crud import BaseCRUD

class UserCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('User')

    async def unique_logic(self, *args, **kwargs):
        # Implement any User-specific logic here
        pass

    async def get_by_field(self, field: str, value):
        db = await self.get_db()
        response = db.table('User').select('*').eq(field, value).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None

    async def user_id_exists(self, user_id):
        """
        Checks if a user with the given user_id exists in the system.
        Returns True if exists, False otherwise.
        """
        db = await self.get_db()
        response = db.table('User').select('id').eq('id', user_id).execute()
        return bool(response.data and len(response.data) > 0)

def serialize_datetimes(data):
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = serialize_datetimes(v)
        elif isinstance(v, list):
            data[k] = [serialize_datetimes(i) if isinstance(i, dict) else i for i in v]
        elif hasattr(v, 'isoformat'):
            data[k] = v.isoformat()
    return data
