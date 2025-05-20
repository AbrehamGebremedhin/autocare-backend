from app.db.crud import BaseCRUD

class UserCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('User')

    async def unique_logic(self, *args, **kwargs):
        # Implement any User-specific logic here
        pass
