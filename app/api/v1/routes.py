from fastapi import APIRouter, Request
from app.api.v1.user_auth import router as auth_router
from app.api.v1.background_tasks import router as background_tasks_router
from app.api.v1.chat_route import router as chat_router
from app.api.v1.car_route import router as user_car_router
from app.api.v1.security_route import router as security_router
from app.api.v1.health import router as health_router
from app.utils.limiter import limiter

router = APIRouter()
router.include_router(auth_router, tags=["Authentication"])
router.include_router(background_tasks_router, tags=["Background Tasks"], prefix="/background-tasks")
router.include_router(chat_router, tags=["Chat"], prefix="/chat")
router.include_router(user_car_router, tags=["Car"], prefix="/car")
router.include_router(security_router, tags=["Security"], prefix="/security")
router.include_router(health_router, tags=["Health"])

# Legacy health check for backward compatibility
@router.get(
    "/health-simple",
    tags=["Health"],
    summary="Simple Health Check",
    description="Simple health check endpoint for backward compatibility."
)
@limiter.limit("10/minute")
async def simple_health_check(request: Request):
    return {
        "status": "ok",
        "success": True
    }