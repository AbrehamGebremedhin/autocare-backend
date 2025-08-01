from fastapi import APIRouter, Request, HTTPException, Depends
from typing import Dict, Any, List, Optional
from app.utils.audit_logging import audit_logger, AuditEventType
from app.utils.limiter import enhanced_limiter
from app.utils.logger import get_logger_instance
from app.utils.auth_middleware import require_role, get_current_user
from app.utils.exceptions import InsufficientPermissionsException
from pydantic import BaseModel

router = APIRouter()
logger = get_logger_instance("SecurityAPI")

class SecurityStatsResponse(BaseModel):
    total_events: int
    events_by_type: Dict[str, int]
    events_by_risk: Dict[str, int]
    unique_users: int
    unique_ips: int
    failed_authentications: int
    security_violations: int
    rate_limit_exceeded: int

class AuditEventResponse(BaseModel):
    timestamp: float
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    endpoint: Optional[str]
    method: Optional[str]
    status_code: Optional[int]
    risk_level: str
    details: Dict[str, Any]

@router.get(
    "/security/stats",
    response_model=SecurityStatsResponse,
    summary="Get Security Statistics",
    description="Retrieve security statistics from audit logs. Requires admin privileges.",
    tags=["Security"]
)
async def get_security_stats(
    request: Request,
    hours: int = 24,
    current_user: Dict[str, Any] = Depends(require_role("admin"))
):
    """Get security statistics from audit logs"""
    await audit_logger.log_event(
        event_type=AuditEventType.ADMIN_ACTION,
        user_id=current_user.get("id"),
        ip_address=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        details={"action": "security_stats_access", "hours": hours}
    )
    
    try:
        stats = await audit_logger.get_security_stats(hours=hours)
        return SecurityStatsResponse(**stats)
    except Exception as e:
        await logger.error(f"Error retrieving security stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving security statistics")

@router.get(
    "/security/events",
    response_model=List[AuditEventResponse],
    summary="Get Recent Security Events",
    description="Retrieve recent security events from audit logs. Requires admin privileges.",
    tags=["Security"]
)
async def get_security_events(
    request: Request,
    event_type: Optional[str] = None,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    hours: int = 24,
    limit: int = 100,
    current_user: Dict[str, Any] = Depends(require_role("admin"))
):
    """Get recent security events"""
    await audit_logger.log_event(
        event_type=AuditEventType.ADMIN_ACTION,
        user_id=current_user.get("id"),
        ip_address=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        details={
            "action": "security_events_access",
            "filters": {
                "event_type": event_type,
                "user_id": user_id,
                "ip_address": ip_address,
                "hours": hours,
                "limit": limit
            }
        }
    )
    
    try:
        # Convert string to enum if provided
        audit_event_type = None
        if event_type:
            try:
                audit_event_type = AuditEventType(event_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
        
        events = await audit_logger.get_recent_events(
            event_type=audit_event_type,
            user_id=user_id,
            ip_address=ip_address,
            hours=hours,
            limit=limit
        )
        
        return [AuditEventResponse(**event) for event in events]
        
    except HTTPException:
        raise
    except Exception as e:
        await logger.error(f"Error retrieving security events: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving security events")

@router.post(
    "/security/test-validation",
    summary="Test Input Validation",
    description="Test endpoint for input validation. Used for security testing.",
    tags=["Security"]
)
async def test_input_validation(
    request: Request,
    test_data: Dict[str, Any]
):
    """Test endpoint for input validation"""
    await audit_logger.log_event(
        event_type=AuditEventType.API_REQUEST,
        ip_address=request.client.host if request.client else None,
        endpoint=request.url.path,
        method=request.method,
        details={"action": "validation_test", "data_keys": list(test_data.keys())}
    )
    
    return {
        "message": "Input validation passed",
        "data_received": {
            "keys": list(test_data.keys()),
            "total_size": len(str(test_data))
        },
        "validation_status": "passed"
    }

@router.get(
    "/security/rate-limit-status",
    summary="Get Rate Limit Status",
    description="Get current rate limit status for the requesting IP/user.",
    tags=["Security"]
)
async def get_rate_limit_status(request: Request):
    """Get rate limit status for current request"""
    ip_address = request.client.host if request.client else None
    
    # This is a simplified status check
    # In a real implementation, you'd query the rate limiter's internal state
    status = {
        "ip_address": ip_address,
        "rate_limit_active": True,
        "requests_remaining": "Unknown",  # Would need to implement in enhanced_limiter
        "reset_time": "Unknown",
        "status": "active"
    }
    
    await audit_logger.log_event(
        event_type=AuditEventType.API_REQUEST,
        ip_address=ip_address,
        endpoint=request.url.path,
        method=request.method,
        details={"action": "rate_limit_status_check"}
    )
    
    return status
