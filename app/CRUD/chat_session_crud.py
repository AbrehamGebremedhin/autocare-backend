from app.db.crud import BaseCRUD

class ChatSessionCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('Chatsession')

    async def unique_logic(self, *args, **kwargs):
        # Implement any Chat_Session-specific logic here
        pass
