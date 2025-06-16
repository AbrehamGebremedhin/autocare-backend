from fastapi import APIRouter
from app.api.v1.user_auth import router as auth_router
from app.api.v1.background_tasks import router as background_tasks_router
from app.api.v1.chat_route import router as chat_router

router = APIRouter()
router.include_router(auth_router)
router.include_router(background_tasks_router)
router.include_router(chat_router)

@router.get("/health")
async def health_check():
    return {"status": "ok"}