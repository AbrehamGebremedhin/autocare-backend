from fastapi import Request, HTTPException
from typing import Any, Dict, List, Optional
import re
import json
from pydantic import BaseModel, ValidationError
from app.utils.logger import get_logger_instance

class SecurityConfig:
    """Security configuration for input validation"""
    
    # Maximum sizes
    MAX_JSON_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_STRING_LENGTH = 10000
    MAX_ARRAY_LENGTH = 1000
    MAX_OBJECT_DEPTH = 10
    
    # Blocked patterns (potential injection attempts)
    BLOCKED_PATTERNS = [
        # XSS patterns
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'on\w+\s*=',  # Event handlers
        r'data:text/html',  # Data URLs
        r'eval\s*\(',  # Code execution
        r'expression\s*\(',  # CSS expression
        
        # Code injection
        r'import\s+',  # Module imports
        r'require\s*\(',  # Node.js requires
        r'(\$\{|\#\{)',  # Template injection
        
        # SQL injection patterns - balanced approach
        r"('\s*(union|select|drop|delete|insert|update|or|and)|;\s*(union|select|drop|delete|insert|update))",  # SQL injection patterns
        r"'\s*(or|and)\s*'[^']*'\s*=\s*'[^']*'",  # Classic OR/AND injection like '1'='1'
        r'\b(union\s+select|drop\s+table|delete\s+from|insert\s+into)\b',  # SQL statement patterns
        r'(--[^\r\n]*|/\*.*?\*/)',  # SQL comments
        
        # Command injection patterns - conservative approach  
        r';\s*(rm|wget|curl|nc|netcat|telnet|ssh)\s',  # Clear command injection
        r'\|\s*(rm|wget|curl|nc|netcat|telnet|ssh)\s',  # Pipe to dangerous commands
        r'&&\s*(rm|wget|curl|nc|netcat|telnet|ssh)\s',  # AND with dangerous commands
        r'\$\([^)]*\)',  # Command substitution
        r'`[^`]+`',  # Backtick execution
        r'(/etc/passwd|/etc/shadow|/proc/|/sys/)',  # Sensitive file access
    ]
    
    # File type restrictions
    ALLOWED_FILE_EXTENSIONS = {'.pdf', '.txt', '.json', '.csv'}
    BLOCKED_FILE_EXTENSIONS = {'.exe', '.bat', '.sh', '.ps1', '.php', '.jsp', '.asp'}

class InputValidator:
    """Enhanced input validation with security checks"""
    
    def __init__(self):
        self.logger = get_logger_instance("InputValidator")
        self.config = SecurityConfig()
    
    async def validate_request(self, request: Request) -> bool:
        """Validate incoming request for security threats"""
        try:
            # Validate URL path
            if not await self._validate_path(request.url.path):
                return False
            
            # Validate query parameters
            if not await self._validate_query_params(dict(request.query_params)):
                return False
            
            # Validate headers
            if not await self._validate_headers(dict(request.headers)):
                return False
            
            # Validate body if present
            if hasattr(request, '_body') and request._body:
                body = await self._get_request_body(request)
                if body and not await self._validate_body(body):
                    return False
            
            return True
            
        except Exception as e:
            await self.logger.error(f"Error during request validation: {str(e)}")
            return False
    
    async def _validate_path(self, path: str) -> bool:
        """Validate URL path for suspicious patterns"""
        if len(path) > 2000:  # Extremely long paths
            await self.logger.warning(f"Rejected request with overly long path: {len(path)} chars")
            return False
        
        # Check for path traversal attempts
        if '../' in path or '..\\' in path:
            await self.logger.warning(f"Path traversal attempt detected: {path}")
            return False
        
        # Check for null bytes
        if '\x00' in path:
            await self.logger.warning(f"Null byte in path detected: {path}")
            return False
        
        return True
    
    async def _validate_query_params(self, params: Dict[str, Any]) -> bool:
        """Validate query parameters"""
        for key, value in params.items():
            if not await self._validate_string_input(str(key)) or \
               not await self._validate_string_input(str(value)):
                await self.logger.warning(f"Malicious query parameter detected: {key}={value}")
                return False
        
        return True
    
    async def _validate_headers(self, headers: Dict[str, str]) -> bool:
        """Validate HTTP headers"""
        dangerous_headers = ['x-forwarded-for', 'x-real-ip']
        host_headers = ['host']
        
        for name, value in headers.items():
            name_lower = name.lower()
            
            # Check header injection
            if '\r' in value or '\n' in value:
                await self.logger.warning(f"Header injection attempt: {name}")
                return False
            
            # Check for extremely long header values
            if len(value) > 8192:
                await self.logger.warning(f"Overly long header value: {name}")
                return False
            
            # Special validation for IP headers
            if name_lower in dangerous_headers:
                if not await self._validate_ip_header(value):
                    return False
            
            # Special validation for host headers (can include port)
            if name_lower in host_headers:
                if not await self._validate_host_header(value):
                    return False
        
        return True
    
    async def _validate_ip_header(self, value: str) -> bool:
        """Validate IP address in headers"""
        # Simple IP validation pattern
        ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        
        # Allow comma-separated IPs for proxy headers
        ips = [ip.strip() for ip in value.split(',')]
        
        for ip in ips:
            if not re.match(ip_pattern, ip) and ip != 'unknown':
                await self.logger.warning(f"Invalid IP in header: {ip}")
                return False
        
        return True

    async def _validate_host_header(self, value: str) -> bool:
        """Validate host header (can include hostname:port)"""
        # Host header pattern: hostname or IP with optional port
        # Supports: localhost, 127.0.0.1, domain.com, 127.0.0.1:8000, localhost:3000
        host_pattern = r'^(?:(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*)(?::[0-9]{1,5})?$'
        
        if not re.match(host_pattern, value):
            await self.logger.warning(f"Invalid host header: {value}")
            return False
        
        return True
    
    async def _get_request_body(self, request: Request) -> Optional[str]:
        """Safely get request body"""
        try:
            if hasattr(request, '_body'):
                body_bytes = request._body
            else:
                body_bytes = await request.body()
            
            if len(body_bytes) > self.config.MAX_JSON_SIZE:
                await self.logger.warning(f"Request body too large: {len(body_bytes)} bytes")
                return None
            
            return body_bytes.decode('utf-8', errors='replace')
        
        except Exception as e:
            await self.logger.error(f"Error reading request body: {str(e)}")
            return None
    
    async def _validate_body(self, body: str) -> bool:
        """Validate request body content"""
        if not await self._validate_string_input(body):
            return False
        
        # Try to parse as JSON if it looks like JSON
        if body.strip().startswith(('{', '[')):
            try:
                data = json.loads(body)
                return await self._validate_json_data(data, depth=0)
            except json.JSONDecodeError:
                # Not valid JSON, but that's okay for non-JSON endpoints
                pass
        
        return True
    
    async def _validate_json_data(self, data: Any, depth: int = 0) -> bool:
        """Recursively validate JSON data"""
        if depth > self.config.MAX_OBJECT_DEPTH:
            await self.logger.warning(f"JSON object too deeply nested: depth {depth}")
            return False
        
        if isinstance(data, dict):
            if len(data) > self.config.MAX_ARRAY_LENGTH:
                await self.logger.warning(f"JSON object has too many keys: {len(data)}")
                return False
            
            for key, value in data.items():
                if not await self._validate_string_input(str(key)):
                    return False
                if not await self._validate_json_data(value, depth + 1):
                    return False
        
        elif isinstance(data, list):
            if len(data) > self.config.MAX_ARRAY_LENGTH:
                await self.logger.warning(f"JSON array too long: {len(data)}")
                return False
            
            for item in data:
                if not await self._validate_json_data(item, depth + 1):
                    return False
        
        elif isinstance(data, str):
            if not await self._validate_string_input(data):
                return False
        
        return True
    
    async def _validate_string_input(self, text: str) -> bool:
        """Validate string input for malicious patterns"""
        if len(text) > self.config.MAX_STRING_LENGTH:
            await self.logger.warning(f"String input too long: {len(text)} chars")
            return False
        
        # Check for blocked patterns
        for pattern in self.config.BLOCKED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                await self.logger.warning(f"Blocked pattern detected: {pattern[:50]}...")
                return False
        
        # Check for control characters (except common whitespace)
        for char in text:
            if ord(char) < 32 and char not in '\t\n\r':
                await self.logger.warning(f"Control character detected: {ord(char)}")
                return False
        
        return True

# Global validator instance
input_validator = InputValidator()

async def validation_middleware(request: Request, call_next):
    """Middleware for input validation and security checks"""
    # Skip validation for certain endpoints
    if request.url.path in ['/health', '/docs', '/redoc', '/openapi.json']:
        response = await call_next(request)
        return response
    
    # Perform validation
    if not await input_validator.validate_request(request):
        raise HTTPException(
            status_code=400,
            detail="Invalid or potentially malicious input detected"
        )
    
    response = await call_next(request)
    return response
