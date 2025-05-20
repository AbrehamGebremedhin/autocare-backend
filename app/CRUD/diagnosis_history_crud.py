from app.db.crud import BaseCRUD

class DiagnosisHistoryCRUD(BaseCRUD):
    def __init__(self):
        super().__init__('Diagnosis_history')

    async def unique_logic(self, *args, **kwargs):
        # Implement any Diagnosis_History-specific logic here
        pass
