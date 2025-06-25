from app.db.crud import BaseCRUD

class UserCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('User')

    async def unique_logic(self, *args, **kwargs):
        # Implement any User-specific logic here
        pass

    async def get_by_field(self, client, field: str, value):
        # Use Supabase Python client to query the user table
        response = client.table('User').select('*').eq(field, value).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None

def serialize_datetimes(data):
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = serialize_datetimes(v)
        elif isinstance(v, list):
            data[k] = [serialize_datetimes(i) if isinstance(i, dict) else i for i in v]
        elif hasattr(v, 'isoformat'):
            data[k] = v.isoformat()
    return data
