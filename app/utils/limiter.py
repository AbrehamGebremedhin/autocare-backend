from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException
import time
import asyncio
from app.utils.logger import get_logger_instance

class EnhancedRateLimiter:
    """Enhanced rate limiter with adaptive limits and user-based tracking"""
    
    def __init__(self):
        self.logger = get_logger_instance("RateLimiter")
        self.user_limits: Dict[str, Dict[str, Any]] = {}
        self.ip_limits: Dict[str, Dict[str, Any]] = {}
        self.global_limits = {
            'requests_per_minute': 300,  # Increased from 100
            'requests_per_hour': 3000,   # Increased from 1000
            'burst_threshold': 100       # Increased from 20
        }
    
    def get_identifier(self, request: Request) -> tuple:
        """Get both IP and user identifier from request"""
        ip = get_remote_address(request)
        user_id = getattr(request.state, 'user_id', None)
        return ip, user_id
    
    async def check_rate_limit(self, request: Request, endpoint: str = None) -> bool:
        """Check if request should be rate limited"""
        ip, user_id = self.get_identifier(request)
        current_time = time.time()
        
        # Check IP-based limits
        if not await self._check_ip_limit(ip, current_time, endpoint):
            await self.logger.warning(f"Rate limit exceeded for IP: {ip}, endpoint: {endpoint}")
            return False
        
        # Check user-based limits if user is identified
        if user_id and not await self._check_user_limit(user_id, current_time, endpoint):
            await self.logger.warning(f"Rate limit exceeded for user: {user_id}, endpoint: {endpoint}")
            return False
        
        return True
    
    async def _check_ip_limit(self, ip: str, current_time: float, endpoint: str = None) -> bool:
        """Check IP-based rate limits"""
        if ip not in self.ip_limits:
            self.ip_limits[ip] = {
                'requests': [],
                'last_request': current_time,
                'burst_count': 0
            }
        
        ip_data = self.ip_limits[ip]
        
        # Clean old requests (older than 1 hour)
        ip_data['requests'] = [req_time for req_time in ip_data['requests'] 
                              if current_time - req_time < 3600]
        
        # Check burst limit (requests in last 10 seconds)
        recent_requests = [req_time for req_time in ip_data['requests'] 
                          if current_time - req_time < 10]
        
        if len(recent_requests) >= self.global_limits['burst_threshold']:
            return False
        
        # Check hourly limit
        if len(ip_data['requests']) >= self.global_limits['requests_per_hour']:
            return False
        
        # Check minute limit
        minute_requests = [req_time for req_time in ip_data['requests'] 
                          if current_time - req_time < 60]
        
        if len(minute_requests) >= self.global_limits['requests_per_minute']:
            return False
        
        # Add current request
        ip_data['requests'].append(current_time)
        ip_data['last_request'] = current_time
        
        return True
    
    async def _check_user_limit(self, user_id: str, current_time: float, endpoint: str = None) -> bool:
        """Check user-based rate limits with adaptive thresholds"""
        if user_id not in self.user_limits:
            self.user_limits[user_id] = {
                'requests': [],
                'last_request': current_time,
                'trust_score': 1.0  # Higher score = more trusted user
            }
        
        user_data = self.user_limits[user_id]
        
        # Clean old requests
        user_data['requests'] = [req_time for req_time in user_data['requests'] 
                                if current_time - req_time < 3600]
        
        # Adaptive limits based on trust score
        user_hourly_limit = int(self.global_limits['requests_per_hour'] * user_data['trust_score'])
        user_minute_limit = int(self.global_limits['requests_per_minute'] * user_data['trust_score'])
        
        # Check limits
        if len(user_data['requests']) >= user_hourly_limit:
            return False
        
        minute_requests = [req_time for req_time in user_data['requests'] 
                          if current_time - req_time < 60]
        
        if len(minute_requests) >= user_minute_limit:
            return False
        
        # Add current request
        user_data['requests'].append(current_time)
        user_data['last_request'] = current_time
        
        # Update trust score based on usage pattern
        await self._update_trust_score(user_id, user_data, current_time)
        
        return True
    
    async def _update_trust_score(self, user_id: str, user_data: Dict[str, Any], current_time: float):
        """Update user trust score based on behavior"""
        # Simple trust score algorithm - can be enhanced
        recent_requests = [req_time for req_time in user_data['requests'] 
                          if current_time - req_time < 300]  # Last 5 minutes
        
        if len(recent_requests) > 50:  # High usage pattern
            user_data['trust_score'] = max(0.5, user_data['trust_score'] - 0.1)
        elif len(recent_requests) < 10:  # Normal usage pattern
            user_data['trust_score'] = min(2.0, user_data['trust_score'] + 0.05)

# Enhanced limiter instance
enhanced_limiter = EnhancedRateLimiter()

# Shared limiter instance for use across the app (backward compatibility)
limiter = Limiter(key_func=get_remote_address)

async def rate_limit_middleware(request: Request, call_next):
    """Middleware for enhanced rate limiting"""
    endpoint = request.url.path
    
    # Skip rate limiting for health checks and static files
    if endpoint in ['/health', '/docs', '/redoc', '/openapi.json']:
        response = await call_next(request)
        return response
    
    # Check rate limits
    if not await enhanced_limiter.check_rate_limit(request, endpoint):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )
    
    response = await call_next(request)
    return response
