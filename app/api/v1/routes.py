from fastapi import APIRouter, Request
from app.api.v1.user_auth import router as auth_router
from app.api.v1.background_tasks import router as background_tasks_router
from app.api.v1.chat_route import router as chat_router
from app.api.v1.car_route import router as user_car_router
from app.utils.limiter import limiter

router = APIRouter()
router.include_router(auth_router)
router.include_router(background_tasks_router)
router.include_router(chat_router)
router.include_router(user_car_router)

@router.get("/health")
@limiter.limit("5/minute")
async def health_check(request: Request):
    data = {
        "status": "ok",
        "success": True
    }
    return data