"""
Unit tests for enhanced rate limiting middleware and implementation.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import time
from datetime import datetime, timedelta
from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException

from app.utils.limiter import (
    EnhancedRateLimiter,
    enhanced_limiter,
    rate_limit_middleware
)
from slowapi.util import get_remote_address


class TestEnhancedRateLimiter:
    """Test enhanced rate limiter implementation"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create enhanced rate limiter instance"""
        return EnhancedRateLimiter()
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.1"
        request.url = MagicMock()
        request.url.path = "/api/v1/test"
        request.state = MagicMock()
        request.state.user_id = None
        return request
    
    def test_initialization(self, rate_limiter):
        """Test rate limiter initialization"""
        assert rate_limiter.user_limits == {}
        assert rate_limiter.ip_limits == {}
        assert rate_limiter.global_limits['requests_per_minute'] == 100
        assert rate_limiter.global_limits['requests_per_hour'] == 1000
        assert rate_limiter.global_limits['burst_threshold'] == 20
    
    def test_get_identifier_ip_only(self, rate_limiter, mock_request):
        """Test getting identifier with IP only"""
        with patch('app.utils.limiter.get_remote_address', return_value="192.168.1.1"):
            ip, user_id = rate_limiter.get_identifier(mock_request)
            
            assert ip == "192.168.1.1"
            assert user_id is None
    
    def test_get_identifier_with_user(self, rate_limiter, mock_request):
        """Test getting identifier with user ID"""
        mock_request.state.user_id = "user123"
        
        with patch('app.utils.limiter.get_remote_address', return_value="192.168.1.1"):
            ip, user_id = rate_limiter.get_identifier(mock_request)
            
            assert ip == "192.168.1.1"
            assert user_id == "user123"
    
    def test_check_rate_limit_first_request(self, rate_limiter, mock_request):
        """Test first request is allowed"""
        with patch('app.utils.limiter.get_remote_address', return_value="192.168.1.1"):
            result = asyncio.run(rate_limiter.check_rate_limit(mock_request))
            
            assert result == True
    
    def test_check_rate_limit_within_limits(self, rate_limiter, mock_request):
        """Test request within rate limits"""
        current_time = time.time()
        
        # Pre-populate with some requests
        rate_limiter.ip_limits["192.168.1.1"] = {
            'requests': [current_time - 30, current_time - 20, current_time - 10],
            'last_request': current_time - 10,
            'burst_count': 0
        }
        
        with patch('app.utils.limiter.get_remote_address', return_value="192.168.1.1"):
            result = asyncio.run(rate_limiter.check_rate_limit(mock_request))
            
            assert result == True
    
    def test_check_rate_limit_burst_exceeded(self, rate_limiter, mock_request):
        """Test request exceeding burst limit"""
        current_time = time.time()
        
        # Create burst of recent requests
        recent_requests = [current_time - i for i in range(21)]  # 21 requests in last seconds
        rate_limiter.ip_limits["192.168.1.1"] = {
            'requests': recent_requests,
            'last_request': current_time - 1,
            'burst_count': 21
        }
        
        with patch('app.utils.limiter.get_remote_address', return_value="192.168.1.1"):
            with patch.object(rate_limiter.logger, 'warning', new_callable=AsyncMock):
                result = asyncio.run(rate_limiter.check_rate_limit(mock_request))
                
                assert result == False
    
    def test_check_rate_limit_hourly_exceeded(self, rate_limiter, mock_request):
        """Test request exceeding hourly limit"""
        current_time = time.time()
        
        # Create 1001 requests in the last hour
        hourly_requests = [current_time - (i * 3) for i in range(1001)]
        rate_limiter.ip_limits["192.168.1.1"] = {
            'requests': hourly_requests,
            'last_request': current_time - 1,
            'burst_count': 0
        }
        
        with patch('app.utils.limiter.get_remote_address', return_value="192.168.1.1"):
            with patch.object(rate_limiter.logger, 'warning', new_callable=AsyncMock):
                result = asyncio.run(rate_limiter.check_rate_limit(mock_request))
                
                assert result == False
    
    def test_check_rate_limit_minute_exceeded(self, rate_limiter, mock_request):
        """Test request exceeding minute limit"""
        current_time = time.time()
        
        # Create 101 requests in the last minute
        minute_requests = [current_time - i for i in range(101)]
        rate_limiter.ip_limits["192.168.1.1"] = {
            'requests': minute_requests,
            'last_request': current_time - 1,
            'burst_count': 0
        }
        
        with patch('app.utils.limiter.get_remote_address', return_value="192.168.1.1"):
            with patch.object(rate_limiter.logger, 'warning', new_callable=AsyncMock):
                result = asyncio.run(rate_limiter.check_rate_limit(mock_request))
                
                assert result == False
    
    def test_check_rate_limit_with_user(self, rate_limiter, mock_request):
        """Test rate limit checking with authenticated user"""
        mock_request.state.user_id = "user123"
        
        with patch('app.utils.limiter.get_remote_address', return_value="192.168.1.1"):
            result = asyncio.run(rate_limiter.check_rate_limit(mock_request))
            
            assert result == True
            assert "user123" in rate_limiter.user_limits
    
    def test_check_user_limit_adaptive_limits(self, rate_limiter, mock_request):
        """Test user rate limits with adaptive thresholds"""
        current_time = time.time()
        user_id = "user123"
        
        # Set up user with high trust score
        rate_limiter.user_limits[user_id] = {
            'requests': [],
            'last_request': current_time,
            'trust_score': 2.0  # High trust
        }
        
        result = asyncio.run(rate_limiter._check_user_limit(user_id, current_time))
        
        assert result == True
        # Trust score should allow more requests
        assert rate_limiter.user_limits[user_id]['trust_score'] == 2.0
    
    def test_check_user_limit_low_trust(self, rate_limiter):
        """Test user rate limits with low trust score"""
        current_time = time.time()
        user_id = "user123"
        
        # Set up user with low trust score and many requests
        rate_limiter.user_limits[user_id] = {
            'requests': [current_time - i for i in range(300)],  # 300 requests
            'last_request': current_time,
            'trust_score': 0.5  # Low trust
        }
        
        result = asyncio.run(rate_limiter._check_user_limit(user_id, current_time))
        
        # Should be denied due to low trust and high usage
        assert result == False
    
    def test_update_trust_score_high_usage(self, rate_limiter):
        """Test trust score update for high usage"""
        current_time = time.time()
        user_id = "user123"
        
        # High usage pattern
        user_data = {
            'requests': [current_time - i for i in range(60)],  # 60 requests in 5 minutes
            'last_request': current_time,
            'trust_score': 1.0
        }
        
        asyncio.run(rate_limiter._update_trust_score(user_id, user_data, current_time))
        
        # Trust score should decrease
        assert user_data['trust_score'] < 1.0
    
    def test_update_trust_score_normal_usage(self, rate_limiter):
        """Test trust score update for normal usage"""
        current_time = time.time()
        user_id = "user123"
        
        # Normal usage pattern
        user_data = {
            'requests': [current_time - i * 60 for i in range(5)],  # 5 requests in 5 minutes
            'last_request': current_time,
            'trust_score': 1.0
        }
        
        asyncio.run(rate_limiter._update_trust_score(user_id, user_data, current_time))
        
        # Trust score should increase
        assert user_data['trust_score'] > 1.0
    
    def test_ip_limit_cleanup(self, rate_limiter):
        """Test cleanup of old IP requests"""
        current_time = time.time()
        old_time = current_time - 7200  # 2 hours ago
        
        # Set up IP with old and new requests
        rate_limiter.ip_limits["192.168.1.1"] = {
            'requests': [old_time, current_time - 30, current_time - 10],
            'last_request': current_time - 10,
            'burst_count': 0
        }
        
        result = asyncio.run(rate_limiter._check_ip_limit("192.168.1.1", current_time))
        
        # Old request should be cleaned up
        requests = rate_limiter.ip_limits["192.168.1.1"]['requests']
        assert len(requests) == 3  # 2 recent + 1 new
        assert old_time not in requests
    
    def test_user_limit_cleanup(self, rate_limiter):
        """Test cleanup of old user requests"""
        current_time = time.time()
        old_time = current_time - 7200  # 2 hours ago
        user_id = "user123"
        
        # Set up user with old and new requests
        rate_limiter.user_limits[user_id] = {
            'requests': [old_time, current_time - 30, current_time - 10],
            'last_request': current_time - 10,
            'trust_score': 1.0
        }
        
        result = asyncio.run(rate_limiter._check_user_limit(user_id, current_time))
        
        # Old request should be cleaned up
        requests = rate_limiter.user_limits[user_id]['requests']
        assert len(requests) == 3  # 2 recent + 1 new
        assert old_time not in requests


class TestRateLimitMiddleware:
    """Test rate limit middleware"""
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.1"
        request.url = MagicMock()
        request.url.path = "/api/v1/test"
        request.state = MagicMock()
        request.state.user_id = None
        return request
    
    def test_middleware_allowed_request(self, mock_request):
        """Test middleware with allowed request"""
        async def mock_call_next(request):
            return Response("OK", status_code=200)
        
        with patch('app.utils.limiter.enhanced_limiter') as mock_limiter:
            mock_limiter.check_rate_limit = AsyncMock(return_value=True)
            
            response = asyncio.run(rate_limit_middleware(mock_request, mock_call_next))
            
            assert response.status_code == 200
            mock_limiter.check_rate_limit.assert_called_once_with(mock_request, "/api/v1/test")
    
    def test_middleware_rate_limited_request(self, mock_request):
        """Test middleware with rate limited request"""
        async def mock_call_next(request):
            return Response("OK", status_code=200)
        
        with patch('app.utils.limiter.enhanced_limiter') as mock_limiter:
            mock_limiter.check_rate_limit = AsyncMock(return_value=False)
            
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(rate_limit_middleware(mock_request, mock_call_next))
            
            assert exc_info.value.status_code == 429
            assert "Rate limit exceeded" in exc_info.value.detail
            assert exc_info.value.headers["Retry-After"] == "60"
    
    def test_middleware_excluded_paths(self):
        """Test middleware with excluded paths"""
        excluded_paths = ['/health', '/docs', '/redoc', '/openapi.json']
        
        for path in excluded_paths:
            request = MagicMock(spec=Request)
            request.url = MagicMock()
            request.url.path = path
            
            async def mock_call_next(request):
                return Response("OK", status_code=200)
            
            with patch('app.utils.limiter.enhanced_limiter') as mock_limiter:
                response = asyncio.run(rate_limit_middleware(request, mock_call_next))
                
                assert response.status_code == 200
                # Rate limiter should not be called for excluded paths
                mock_limiter.check_rate_limit.assert_not_called()


class TestGlobalRateLimiterInstance:
    """Test global rate limiter instance"""
    
    def test_global_instance_exists(self):
        """Test that global enhanced_limiter instance exists"""
        assert enhanced_limiter is not None
        assert isinstance(enhanced_limiter, EnhancedRateLimiter)
    
    def test_global_instance_configuration(self):
        """Test global instance has correct configuration"""
        assert enhanced_limiter.global_limits['requests_per_minute'] == 100
        assert enhanced_limiter.global_limits['requests_per_hour'] == 1000
        assert enhanced_limiter.global_limits['burst_threshold'] == 20


class TestRateLimitIntegration:
    """Integration tests for rate limiting"""
    
    def test_complete_rate_limit_flow(self):
        """Test complete rate limiting flow"""
        limiter = EnhancedRateLimiter()
        
        # Override limits for testing
        limiter.global_limits['requests_per_minute'] = 2
        limiter.global_limits['burst_threshold'] = 3
        
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        request.url = MagicMock()
        request.url.path = "/api/test"
        request.state = MagicMock()
        request.state.user_id = None
        
        async def test_flow():
            with patch('app.utils.limiter.get_remote_address', return_value="192.168.1.100"):
                # First requests should be allowed
                assert await limiter.check_rate_limit(request) == True
                assert await limiter.check_rate_limit(request) == True
                
                # Third request should be denied (exceeds minute limit)
                assert await limiter.check_rate_limit(request) == False
        
        asyncio.run(test_flow())
    
    def test_middleware_integration(self):
        """Test middleware integration with rate limiter"""
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.1"
        request.url = MagicMock()
        request.url.path = "/api/test"
        request.state = MagicMock()
        request.state.user_id = None
        
        async def mock_call_next(request):
            return Response("OK", status_code=200)
        
        # Create temporary limiter with low limits
        temp_limiter = EnhancedRateLimiter()
        temp_limiter.global_limits['requests_per_minute'] = 1
        
        with patch('app.utils.limiter.enhanced_limiter', temp_limiter):
            with patch('app.utils.limiter.get_remote_address', return_value="192.168.1.1"):
                # First request should be allowed
                response1 = asyncio.run(rate_limit_middleware(request, mock_call_next))
                assert response1.status_code == 200
                
                # Second request should be rate limited
                with pytest.raises(HTTPException) as exc_info:
                    asyncio.run(rate_limit_middleware(request, mock_call_next))
                
                assert exc_info.value.status_code == 429
    
    def test_user_based_rate_limiting(self):
        """Test user-based rate limiting functionality"""
        limiter = EnhancedRateLimiter()
        limiter.global_limits['requests_per_minute'] = 2
        
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "192.168.1.1"
        request.url = MagicMock()
        request.url.path = "/api/test"
        request.state = MagicMock()
        request.state.user_id = "test_user"
        
        async def test_user_flow():
            with patch('app.utils.limiter.get_remote_address', return_value="192.168.1.1"):
                # First requests should be allowed
                assert await limiter.check_rate_limit(request) == True
                assert await limiter.check_rate_limit(request) == True
                
                # Check that user limits are tracked
                assert "test_user" in limiter.user_limits
                assert limiter.user_limits["test_user"]["trust_score"] == 1.0
        
        asyncio.run(test_user_flow())
