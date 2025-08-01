from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from app.db.base import SupabaseDBHandler, get_db_handler
from app.utils.redis_cache import get_redis_cache, RedisCache
from app.utils.logger import get_logger_instance
from datetime import datetime
import asyncio

router = APIRouter()
logger = get_logger_instance("HealthCheck")

@router.get("/health", tags=["Health"])
async def health_check():
    """
    Basic health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "autocare-backend"
    }

@router.get("/health/detailed", tags=["Health"])
async def detailed_health_check(
    db_handler: SupabaseDBHandler = Depends(get_db_handler),
    redis_cache: RedisCache = Depends(get_redis_cache)
):
    """
    Detailed health check with service dependencies
    """
    start_time = datetime.utcnow()
    health_status = {
        "status": "healthy",
        "timestamp": start_time.isoformat(),
        "service": "autocare-backend",
        "version": "1.0.0",
        "checks": {}
    }
    
    # Database health check
    try:
        db_healthy = await db_handler.health_check()
        health_status["checks"]["database"] = {
            "status": "healthy" if db_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat()
        }
        if db_healthy:
            db_stats = await db_handler.get_connection_stats()
            health_status["checks"]["database"]["stats"] = db_stats
    except Exception as e:
        await logger.error(f"Database health check failed: {str(e)}")
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": "Connection failed",
            "timestamp": datetime.utcnow().isoformat()
        }
        health_status["status"] = "degraded"
    
    # Redis health check
    try:
        redis_healthy = await redis_cache.health_check()
        health_status["checks"]["redis"] = {
            "status": "healthy" if redis_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat()
        }
        if redis_healthy:
            redis_stats = await redis_cache.get_stats()
            health_status["checks"]["redis"]["stats"] = redis_stats
    except Exception as e:
        await logger.error(f"Redis health check failed: {str(e)}")
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "error": "Connection failed",
            "timestamp": datetime.utcnow().isoformat()
        }
        if health_status["status"] == "healthy":
            health_status["status"] = "degraded"
    
    # Calculate response time
    end_time = datetime.utcnow()
    response_time = (end_time - start_time).total_seconds() * 1000
    health_status["response_time_ms"] = round(response_time, 2)
    
    # Determine overall status
    failed_checks = [check for check in health_status["checks"].values() if check["status"] == "unhealthy"]
    if failed_checks:
        if len(failed_checks) == len(health_status["checks"]):
            health_status["status"] = "unhealthy"
        else:
            health_status["status"] = "degraded"
    
    # Return appropriate HTTP status
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    elif health_status["status"] == "degraded":
        raise HTTPException(status_code=200, detail=health_status)  # Still return 200 for degraded
    
    return health_status

@router.get("/health/ready", tags=["Health"])
async def readiness_check():
    """
    Kubernetes readiness probe endpoint
    """
    try:
        # Quick checks for essential services
        tasks = [
            check_database_quick(),
            check_redis_quick()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # If any check fails, service is not ready
        for result in results:
            if isinstance(result, Exception) or result is False:
                raise HTTPException(
                    status_code=503, 
                    detail={"status": "not ready", "timestamp": datetime.utcnow().isoformat()}
                )
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={"status": "not ready", "error": str(e), "timestamp": datetime.utcnow().isoformat()}
        )

@router.get("/health/live", tags=["Health"])
async def liveness_check():
    """
    Kubernetes liveness probe endpoint
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }

async def check_database_quick() -> bool:
    """Quick database connectivity check"""
    try:
        db_handler = SupabaseDBHandler()
        return await db_handler.health_check()
    except Exception:
        return False

async def check_redis_quick() -> bool:
    """Quick Redis connectivity check"""
    try:
        from app.utils.redis_cache import redis_cache
        return await redis_cache.health_check()
    except Exception:
        return False
