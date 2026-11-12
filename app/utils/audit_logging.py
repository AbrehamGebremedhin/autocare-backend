import json
import time
from typing import Dict, Any, Optional
from enum import Enum
from app.utils.logger import get_logger_instance

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
    USER_CREATED = "user_created"

class AuditLogger:
    """Structured audit logging. Never records request bodies, headers or tokens."""
    
    def __init__(self):
        self.logger = get_logger_instance("AuditLogger")

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
        }
        
        # Log to structured logger
        await self._log_to_file(audit_record)
    
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
    

# Global audit logger instance
audit_logger = AuditLogger()
