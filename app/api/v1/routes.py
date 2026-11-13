from fastapi import APIRouter
from app.api.v1.user_auth import router as auth_router
from app.api.v1.car_route import router as user_car_router
from app.api.v1.session_route import router as session_router
from app.api.v1.finding_route import router as finding_router
from app.api.v1.visual_route import cars_router as visual_cars_router, files_router as visual_files_router

router = APIRouter()
router.include_router(auth_router, tags=["Authentication"])
router.include_router(user_car_router)
router.include_router(session_router)
router.include_router(finding_router)
router.include_router(visual_cars_router)
router.include_router(visual_files_router)
