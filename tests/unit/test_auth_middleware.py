"""
Unit tests for authentication middleware and JWT handling.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
import os

from app.utils.auth_middleware import (
    JWTHandler, 
    access_claims,
    get_current_user, 
    AuthenticationError,
)


class TestJWTHandler:
    """Test JWT token handling"""
    
    @pytest.fixture
    def jwt_handler(self):
        """Create JWT handler instance"""
        return JWTHandler()
    
    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for token creation"""
        return {
            "sub": "user-123",
            "email": "test@example.com",
            "role": "user",
            "permissions": ["read", "write"]
        }
    
    def test_create_access_token(self, jwt_handler, sample_user_data):
        """Test access token creation"""
        token = jwt_handler.create_access_token(sample_user_data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify token
        payload = jwt.decode(token, jwt_handler.secret_key, algorithms=[jwt_handler.algorithm])
        assert payload["sub"] == "user-123"
        assert payload["email"] == "test@example.com"
        assert payload["type"] == "access"
        assert "exp" in payload
    
    def test_create_refresh_token(self, jwt_handler, sample_user_data):
        """Test refresh token creation"""
        token = jwt_handler.create_refresh_token(sample_user_data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify token
        payload = jwt.decode(token, jwt_handler.secret_key, algorithms=[jwt_handler.algorithm])
        assert payload["sub"] == "user-123"
        assert payload["type"] == "refresh"

    def test_tokens_carry_a_fractional_iat(self, jwt_handler, sample_user_data):
        """iat keeps sub-second precision so revocation (users.tokens_valid_after) is exact."""
        import time

        before = time.time()
        for token in (jwt_handler.create_access_token(sample_user_data), jwt_handler.create_refresh_token(sample_user_data)):
            payload = jwt.decode(token, jwt_handler.secret_key, algorithms=[jwt_handler.algorithm])
            assert isinstance(payload["iat"], float) and before <= payload["iat"] <= time.time()
    
    def test_verify_valid_token(self, jwt_handler, sample_user_data):
        """Test verification of valid token"""
        token = jwt_handler.create_access_token(sample_user_data)
        payload = jwt_handler.verify_token(token)
        
        assert payload["sub"] == "user-123"
        assert payload["email"] == "test@example.com"
        assert payload["type"] == "access"
    
    def test_verify_expired_token(self, jwt_handler, sample_user_data):
        """Test verification of expired token"""
        # Create token with past expiration
        past_data = sample_user_data.copy()
        past_exp = datetime.utcnow() - timedelta(hours=1)
        past_data["exp"] = past_exp.timestamp()
        
        token = jwt.encode(past_data, jwt_handler.secret_key, algorithm=jwt_handler.algorithm)
        
        with pytest.raises(AuthenticationError) as exc_info:
            jwt_handler.verify_token(token)
        
        assert "expired" in str(exc_info.value.detail).lower()
    
    def test_verify_invalid_token(self, jwt_handler):
        """Test verification of invalid token"""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(AuthenticationError) as exc_info:
            jwt_handler.verify_token(invalid_token)
        
        assert "invalid" in str(exc_info.value.detail).lower()
    
    def test_verify_token_wrong_secret(self, jwt_handler, sample_user_data):
        """Test verification with wrong secret"""
        # Create token with different secret
        wrong_token = jwt.encode(
            sample_user_data, 
            "wrong-secret-that-is-32-bytes-long", 
            algorithm=jwt_handler.algorithm
        )
        
        with pytest.raises(AuthenticationError):
            jwt_handler.verify_token(wrong_token)


class TestAuthenticationMiddleware:
    """Test authentication middleware functions"""
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = MagicMock(spec=Request)
        request.client.host = "127.0.0.1"
        request.url.path = "/api/v1/test"
        request.method = "GET"
        request.state = MagicMock()
        return request
    
    @pytest.fixture
    def valid_credentials(self):
        """Create valid HTTP authorization credentials"""
        # Use the test environment JWT secret directly
        import os
        jwt_handler = JWTHandler()
        # Ensure we're using the test secret from environment
        jwt_handler.secret_key = os.getenv("JWT_SECRET_KEY", "test-jwt-secret-key-for-unit-tests-must-be-32-chars-minimum")
        
        user_data = {
            "sub": "user-123",
            "email": "test@example.com",
            "role": "user",
            "permissions": ["read", "write"]
        }
        token = jwt_handler.create_access_token(user_data)
        
        credentials = MagicMock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = token
        return credentials
    
    @pytest.fixture
    def invalid_credentials(self):
        """Create invalid HTTP authorization credentials"""
        credentials = MagicMock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = "invalid.token.here"
        return credentials
    
    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self, mock_request, valid_credentials):
        """Test getting current user with valid token"""
        import os
        from app.utils.auth_middleware import jwt_handler
        
        # Patch the global JWT handler to use test environment secret
        original_secret = jwt_handler.secret_key
        jwt_handler.secret_key = os.getenv("JWT_SECRET_KEY", "test-jwt-secret-key-for-unit-tests-must-be-32-chars-minimum")
        
        try:
            with patch('app.utils.auth_middleware.validate_user', new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = True
                
                user = await get_current_user(mock_request, valid_credentials)
                
                assert user["id"] == "user-123"
                assert user["email"] == "test@example.com"
                assert user["role"] == "user"
                assert "read" in user["permissions"]
        finally:
            # Restore original secret
            jwt_handler.secret_key = original_secret
    
    @pytest.mark.asyncio
    async def test_get_current_user_no_credentials(self, mock_request):
        """Test getting current user without credentials"""
        with pytest.raises(AuthenticationError) as exc_info:
            await get_current_user(mock_request, None)
        
        assert "missing" in str(exc_info.value.detail).lower()
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, mock_request, invalid_credentials):
        """Test getting current user with invalid token"""
        with pytest.raises(AuthenticationError):
            await get_current_user(mock_request, invalid_credentials)
    
    @pytest.mark.asyncio
    async def test_get_current_user_refresh_token(self, mock_request):
        """Test rejection of refresh token for access"""
        # Use the test environment JWT secret directly
        import os
        from app.utils.auth_middleware import jwt_handler as global_jwt_handler
        
        jwt_handler = JWTHandler()
        jwt_handler.secret_key = os.getenv("JWT_SECRET_KEY", "test-jwt-secret-key-for-unit-tests-must-be-32-chars-minimum")
        
        # Patch the global JWT handler to use test environment secret
        original_secret = global_jwt_handler.secret_key
        global_jwt_handler.secret_key = jwt_handler.secret_key
        
        try:
            user_data = {"sub": "user-123", "email": "test@example.com"}
            refresh_token = jwt_handler.create_refresh_token(user_data)
            
            credentials = MagicMock(spec=HTTPAuthorizationCredentials)
            credentials.credentials = refresh_token
            
            with pytest.raises(AuthenticationError) as exc_info:
                await get_current_user(mock_request, credentials)
            
            assert "invalid token type" in str(exc_info.value.detail).lower()
        finally:
            # Restore original secret
            global_jwt_handler.secret_key = original_secret
    
    @pytest.mark.asyncio
    async def test_get_current_user_missing_subject(self, mock_request):
        """Test handling of token without subject"""
        # Use the test environment JWT secret directly
        import os
        jwt_handler = JWTHandler()
        jwt_handler.secret_key = os.getenv("JWT_SECRET_KEY", "test-jwt-secret-key-for-unit-tests-must-be-32-chars-minimum")
        
        user_data = {"email": "test@example.com", "type": "access"}  # No 'sub'
        token = jwt.encode(
            {**user_data, "exp": datetime.utcnow() + timedelta(minutes=30)},
            jwt_handler.secret_key,
            algorithm=jwt_handler.algorithm
        )
        
        credentials = MagicMock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = token
        
        with pytest.raises(AuthenticationError):
            await get_current_user(mock_request, credentials)
    


class TestValidateUser:
    """The token's sub must still be a row in the users table."""

    USER = "11111111-1111-4111-8111-111111111111"

    @pytest.mark.asyncio
    async def test_existing_user_passes(self, fake_users):
        import time
        from app.utils.auth_middleware import validate_user

        user = fake_users.add("driver@example.com", google_sub="sub-1")
        assert await validate_user(user["id"], time.time()) is True

    @pytest.mark.asyncio
    async def test_token_issued_before_revocation_is_401(self, fake_users):
        from app.utils.auth_middleware import validate_user

        user = fake_users.add("driver@example.com", google_sub="sub-1")
        with pytest.raises(AuthenticationError):
            await validate_user(user["id"], user["tokens_valid_after"] - 0.000001)

    @pytest.mark.asyncio
    async def test_token_without_iat_is_401(self, mock_request, fake_users):
        from app.utils.auth_middleware import jwt_handler

        user = fake_users.add("driver@example.com", google_sub="sub-1")
        token = jwt.encode(
            {"sub": user["id"], "type": "access", "exp": datetime.utcnow() + timedelta(minutes=5)},
            jwt_handler.secret_key, algorithm=jwt_handler.algorithm,
        )
        with pytest.raises(AuthenticationError):
            await get_current_user(mock_request, HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    @pytest.mark.asyncio
    async def test_unknown_user_is_401(self, fake_users):
        from app.utils.auth_middleware import validate_user

        with pytest.raises(AuthenticationError) as exc_info:
            await validate_user(self.USER, 1.0)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_lookup_is_one_bound_query(self, fake_pool):
        from app.utils.auth_middleware import validate_user

        fake_pool.value = True
        assert await validate_user(self.USER, 1_800_000_000.25) is True
        [(method, sql, args)] = fake_pool.calls
        assert method == "fetchval" and "users" in sql and args == (self.USER, 1_800_000_000.25)

    @pytest.mark.asyncio
    async def test_non_uuid_sub_is_401_without_a_query(self, fake_pool):
        from app.utils.auth_middleware import validate_user

        with pytest.raises(AuthenticationError):
            await validate_user("user-123", 1.0)
        assert fake_pool.calls == []

    @pytest.mark.asyncio
    async def test_database_failure_is_503_not_401(self, fake_pool):
        # A 401 would make clients throw their tokens away during a database outage.
        from app.utils.auth_middleware import validate_user
        from app.utils.exceptions import ServiceUnavailableException

        fake_pool.error = ConnectionError("db down")
        with pytest.raises(ServiceUnavailableException) as exc:
            await validate_user(self.USER, 1.0)
        assert exc.value.status_code == 503


class TestProtectedRoutesWithRealValidator:
    """An app token reaches protected routes only while its user exists."""

    def test_existing_user_reaches_protected_route(self, client, fake_users, monkeypatch):
        from app.utils.auth_middleware import jwt_handler
        from app.api.v1 import car_route

        user = fake_users.add("driver@example.com", google_sub="sub-1")
        monkeypatch.setattr(car_route.user_profile_service, "get_user_cars", AsyncMock(return_value=[]))

        token = jwt_handler.create_access_token({"sub": user["id"]})
        r = client.get("/api/v1/cars", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_deleted_user_gets_401_on_protected_route(self, client, fake_users):
        from app.utils.auth_middleware import jwt_handler

        user = fake_users.add("driver@example.com", google_sub="sub-1")
        token = jwt_handler.create_access_token({"sub": user["id"]})
        del fake_users.rows[user["id"]]
        r = client.get("/api/v1/cars", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401

    def test_database_outage_gets_generic_503_on_protected_route(self, client, fake_pool):
        from app.utils.auth_middleware import jwt_handler

        token = jwt_handler.create_access_token({"sub": "11111111-1111-4111-8111-111111111111"})
        fake_pool.error = ConnectionError("connection refused to 10.0.0.5")
        r = client.get("/api/v1/cars", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 503
        body = r.json()
        assert body["error_code"] == "SYS_002" and body["message"] == "Service temporarily unavailable"
        assert "10.0.0.5" not in r.text and "WWW-Authenticate" not in r.headers


class TestErrorLogsHideDetails:
    """Error logs carry only the exception class, never its message."""

    @pytest.mark.asyncio
    async def test_user_validation_logs_class_only(self, fake_pool, caplog):
        import logging
        from app.utils.auth_middleware import validate_user

        from app.utils.exceptions import ServiceUnavailableException

        fake_pool.error = RuntimeError("leaked eyJ-token-body")

        with caplog.at_level(logging.DEBUG), pytest.raises(ServiceUnavailableException):
            await validate_user("11111111-1111-4111-8111-111111111111", 1.0)
        assert "RuntimeError" in caplog.text
        assert "eyJ-token-body" not in caplog.text

    @pytest.mark.asyncio
    async def test_get_current_user_unexpected_error_logs_class_only(self, mock_request, caplog):
        import logging
        from app.utils.auth_middleware import get_current_user

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tok")
        with patch('app.utils.auth_middleware.jwt_handler') as h:
            h.verify_token.side_effect = ValueError("secret internals")
            with caplog.at_level(logging.DEBUG), pytest.raises(AuthenticationError):
                await get_current_user(mock_request, creds)
        assert "ValueError" in caplog.text
        assert "secret internals" not in caplog.text


class TestAccessClaims:
    """access_claims: the token checks of get_current_user, without the database."""

    def _handler(self):
        return JWTHandler()

    def test_valid_access_token_returns_its_payload(self):
        token = self._handler().create_access_token({"sub": "u1", "email": "a@b.c"})
        assert access_claims(token)["sub"] == "u1"

    @pytest.mark.parametrize("token", [None, ""])
    def test_missing_token(self, token):
        with pytest.raises(AuthenticationError) as exc:
            access_claims(token)
        assert exc.value.detail == "Missing authentication token"

    def test_expired_token(self):
        token = self._handler().create_access_token({"sub": "u1"}, issued_at=datetime.now().timestamp() - 7200)
        with pytest.raises(AuthenticationError) as exc:
            access_claims(token)
        assert exc.value.detail == "Token has expired"

    def test_garbage_token(self):
        with pytest.raises(AuthenticationError) as exc:
            access_claims("not-a-jwt")
        assert exc.value.detail == "Invalid token"

    def test_refresh_token_is_the_wrong_type(self):
        with pytest.raises(AuthenticationError) as exc:
            access_claims(self._handler().create_refresh_token({"sub": "u1"}))
        assert exc.value.detail == "Invalid token type"

    @pytest.mark.parametrize("data", [{}, {"sub": ""}])
    def test_token_without_a_subject_is_an_invalid_payload(self, data):
        with pytest.raises(AuthenticationError) as exc:
            access_claims(self._handler().create_access_token(data))
        assert exc.value.detail == "Invalid token payload"

    def test_token_with_a_boolean_iat_is_an_invalid_payload(self):
        h = self._handler()
        token = jwt.encode({"sub": "u1", "type": "access", "iat": True}, h.secret_key, algorithm=h.algorithm)
        with pytest.raises(AuthenticationError) as exc:
            access_claims(token)
        assert exc.value.detail == "Invalid token payload"
