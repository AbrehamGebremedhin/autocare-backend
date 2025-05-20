from app.db.crud import BaseCRUD

class GroundKnowledgeCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('Groundknowledge')

    async def unique_logic(self, *args, **kwargs):
        # Implement any Ground_Knowledge-specific logic here
        pass
