import json
import time
from typing import Dict, Any, Optional, List
from fastapi import Request, Response
from enum import Enum
from app.utils.logger import get_logger_instance
from app.utils.redis_cache import redis_cache
import asyncio

class AuditEventType(Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    API_REQUEST = "api_request"
    DATA_ACCESS = "data_access"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"
    ADMIN_ACTION = "admin_action"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    AUTHENTICATION_FAILED = "authentication_failed"
    DATA_MODIFICATION = "data_modification"

class AuditLogger:
    """Enhanced audit logging system with Redis caching and structured logging"""
    
    def __init__(self):
        self.logger = get_logger_instance("AuditLogger")
        self.redis_cache = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection for audit log caching"""
        try:
            self.redis_cache = redis_cache
        except Exception as e:
            # Fallback to local logging if Redis is not available
            asyncio.create_task(self.logger.warning(f"Redis not available for audit logging: {str(e)}"))
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_level: str = "low"
    ):
        """Log an audit event with structured data"""
        timestamp = time.time()
        
        audit_record = {
            "timestamp": timestamp,
            "event_type": event_type.value,
            "user_id": user_id,
            "ip_address": ip_address,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "risk_level": risk_level,
            "details": details or {},
            "session_id": getattr(details, 'session_id', None) if details else None
        }
        
        # Log to structured logger
        await self._log_to_file(audit_record)
        
        # Cache in Redis for real-time monitoring
        await self._cache_audit_event(audit_record)
        
        # Alert on high-risk events
        if risk_level in ["high", "critical"]:
            await self._handle_high_risk_event(audit_record)
    
    async def _log_to_file(self, audit_record: Dict[str, Any]):
        """Log audit record to file with structured format"""
        log_line = json.dumps(audit_record, separators=(',', ':'))
        
        if audit_record.get("risk_level") == "critical":
            await self.logger.critical(f"AUDIT: {log_line}")
        elif audit_record.get("risk_level") == "high":
            await self.logger.error(f"AUDIT: {log_line}")
        elif audit_record.get("risk_level") == "medium":
            await self.logger.warning(f"AUDIT: {log_line}")
        else:
            await self.logger.info(f"AUDIT: {log_line}")
    
    async def _cache_audit_event(self, audit_record: Dict[str, Any]):
        """Cache audit event in Redis for real-time monitoring"""
        if not self.redis_cache:
            return
        
        try:
            # Store in multiple Redis structures for different query patterns
            timestamp = int(audit_record["timestamp"])
            event_type = audit_record["event_type"]
            user_id = audit_record.get("user_id")
            ip_address = audit_record.get("ip_address")
            
            # Recent events (last 24 hours)
            recent_key = f"audit:recent:{timestamp // 3600}"  # Hour buckets
            await self.redis_cache.lpush(recent_key, json.dumps(audit_record))
            await self.redis_cache.expire(recent_key, 86400)  # 24 hours
            
            # Events by type
            type_key = f"audit:type:{event_type}:{timestamp // 3600}"
            await self.redis_cache.lpush(type_key, json.dumps(audit_record))
            await self.redis_cache.expire(type_key, 86400)
            
            # Events by user
            if user_id:
                user_key = f"audit:user:{user_id}:{timestamp // 3600}"
                await self.redis_cache.lpush(user_key, json.dumps(audit_record))
                await self.redis_cache.expire(user_key, 86400)
            
            # Events by IP
            if ip_address:
                ip_key = f"audit:ip:{ip_address}:{timestamp // 3600}"
                await self.redis_cache.lpush(ip_key, json.dumps(audit_record))
                await self.redis_cache.expire(ip_key, 86400)
                
        except Exception as e:
            await self.logger.error(f"Failed to cache audit event: {str(e)}")
    
    async def _handle_high_risk_event(self, audit_record: Dict[str, Any]):
        """Handle high-risk security events"""
        event_type = audit_record["event_type"]
        risk_level = audit_record["risk_level"]
        
        # Log critical alert
        await self.logger.critical(
            f"HIGH RISK AUDIT EVENT: {event_type} - "
            f"Risk Level: {risk_level} - "
            f"Details: {json.dumps(audit_record)}"
        )
        
        # Additional alerting mechanisms could be added here:
        # - Send to security team
        # - Trigger automated responses
        # - Update threat intelligence
    
    async def get_recent_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve recent audit events from cache"""
        if not self.redis_cache:
            return []
        
        try:
            current_hour = int(time.time()) // 3600
            events = []
            
            for hour_offset in range(hours):
                hour_bucket = current_hour - hour_offset
                
                # Build appropriate key based on filters
                if event_type:
                    key = f"audit:type:{event_type.value}:{hour_bucket}"
                elif user_id:
                    key = f"audit:user:{user_id}:{hour_bucket}"
                elif ip_address:
                    key = f"audit:ip:{ip_address}:{hour_bucket}"
                else:
                    key = f"audit:recent:{hour_bucket}"
                
                # Get events from this hour bucket
                cached_events = await self.redis_cache.lrange(key, 0, limit)
                
                for event_json in cached_events:
                    try:
                        event = json.loads(event_json)
                        events.append(event)
                        if len(events) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
                
                if len(events) >= limit:
                    break
            
            # Sort by timestamp (most recent first)
            events.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            return events[:limit]
            
        except Exception as e:
            await self.logger.error(f"Failed to retrieve audit events: {str(e)}")
            return []
    
    async def get_security_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get security statistics from audit logs"""
        events = await self.get_recent_events(hours=hours, limit=10000)
        
        stats = {
            "total_events": len(events),
            "events_by_type": {},
            "events_by_risk": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "unique_users": set(),
            "unique_ips": set(),
            "failed_authentications": 0,
            "security_violations": 0,
            "rate_limit_exceeded": 0
        }
        
        for event in events:
            # Count by type
            event_type = event.get("event_type", "unknown")
            stats["events_by_type"][event_type] = stats["events_by_type"].get(event_type, 0) + 1
            
            # Count by risk level
            risk_level = event.get("risk_level", "low")
            stats["events_by_risk"][risk_level] += 1
            
            # Track unique users and IPs
            if event.get("user_id"):
                stats["unique_users"].add(event["user_id"])
            if event.get("ip_address"):
                stats["unique_ips"].add(event["ip_address"])
            
            # Count specific security events
            if event_type == AuditEventType.AUTHENTICATION_FAILED.value:
                stats["failed_authentications"] += 1
            elif event_type == AuditEventType.SECURITY_VIOLATION.value:
                stats["security_violations"] += 1
            elif event_type == AuditEventType.RATE_LIMIT_EXCEEDED.value:
                stats["rate_limit_exceeded"] += 1
        
        # Convert sets to counts
        stats["unique_users"] = len(stats["unique_users"])
        stats["unique_ips"] = len(stats["unique_ips"])
        
        return stats

# Global audit logger instance
audit_logger = AuditLogger()

async def audit_middleware(request: Request, call_next):
    """Middleware for audit logging"""
    start_time = time.time()
    user_id = getattr(request.state, 'user_id', None)
    ip_address = request.client.host if request.client else None
    
    # Log API request
    await audit_logger.log_event(
        event_type=AuditEventType.API_REQUEST,
        user_id=user_id,
        ip_address=ip_address,
        endpoint=request.url.path,
        method=request.method,
        details={
            "query_params": dict(request.query_params),
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type")
        }
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Log successful response
        processing_time = time.time() - start_time
        risk_level = "medium" if response.status_code >= 400 else "low"
        
        await audit_logger.log_event(
            event_type=AuditEventType.API_REQUEST,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            risk_level=risk_level,
            details={
                "processing_time": processing_time,
                "response_size": response.headers.get("content-length")
            }
        )
        
        return response
        
    except Exception as e:
        # Log error
        await audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=request.url.path,
            method=request.method,
            risk_level="high",
            details={
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        )
        raise
