from fastapi import APIRouter
from app.api.v1.user_auth import router as auth_router
from app.api.v1.car_route import router as user_car_router

router = APIRouter()
router.include_router(auth_router, tags=["Authentication"])
router.include_router(user_car_router)
