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
    get_current_user, 
    get_current_active_user,
    require_permissions,
    require_role,
    AuthenticationError,
    AuthorizationError
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
            "wrong-secret", 
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
            with patch('app.utils.auth_middleware.validate_user_with_supabase', new_callable=AsyncMock) as mock_validate:
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
    
    @pytest.mark.asyncio
    async def test_get_current_active_user(self, mock_request, valid_credentials):
        """Test getting current active user"""
        import os
        from app.utils.auth_middleware import jwt_handler
        
        # Patch the global JWT handler to use test environment secret
        original_secret = jwt_handler.secret_key
        jwt_handler.secret_key = os.getenv("JWT_SECRET_KEY", "test-jwt-secret-key-for-unit-tests-must-be-32-chars-minimum")
        
        try:
            with patch('app.utils.auth_middleware.validate_user_with_supabase', new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = True
                
                user = await get_current_user(mock_request, valid_credentials)
                active_user = await get_current_active_user(user)
                
                assert active_user == user
        finally:
            # Restore original secret
            jwt_handler.secret_key = original_secret


class TestAuthorizationDecorators:
    """Test authorization decorator functions"""
    
    @pytest.fixture
    def user_with_permissions(self):
        """User with specific permissions"""
        return {
            "id": "user-123",
            "email": "test@example.com",
            "role": "user",
            "permissions": ["read", "write", "delete"]
        }
    
    @pytest.fixture
    def admin_user(self):
        """Admin user"""
        return {
            "id": "admin-123",
            "email": "admin@example.com",
            "role": "admin",
            "permissions": ["admin"]
        }
    
    @pytest.fixture
    def basic_user(self):
        """Basic user with limited permissions"""
        return {
            "id": "basic-123",
            "email": "basic@example.com",
            "role": "user",
            "permissions": ["read"]
        }
    
    def test_require_permissions_success(self, user_with_permissions):
        """Test successful permission check"""
        permission_checker = require_permissions(["read", "write"])
        
        # Should not raise exception
        result = permission_checker(user_with_permissions)
        assert result == user_with_permissions
    
    def test_require_permissions_admin_bypass(self, admin_user):
        """Test that admin bypasses permission checks"""
        permission_checker = require_permissions(["super_secret_permission"])
        
        # Admin should bypass permission check
        result = permission_checker(admin_user)
        assert result == admin_user
    
    def test_require_permissions_failure(self, basic_user):
        """Test failed permission check"""
        permission_checker = require_permissions(["admin", "delete"])
        
        with pytest.raises(AuthorizationError) as exc_info:
            permission_checker(basic_user)
        
        assert "required permissions" in str(exc_info.value.detail).lower()
    
    def test_require_role_success(self, admin_user):
        """Test successful role check"""
        role_checker = require_role("admin")
        
        result = role_checker(admin_user)
        assert result == admin_user
    
    def test_require_role_admin_bypass(self, admin_user):
        """Test that admin can access any role"""
        role_checker = require_role("user")
        
        result = role_checker(admin_user)
        assert result == admin_user
    
    def test_require_role_failure(self, basic_user):
        """Test failed role check"""
        role_checker = require_role("admin")
        
        with pytest.raises(AuthorizationError) as exc_info:
            role_checker(basic_user)
        
        assert "required role" in str(exc_info.value.detail).lower()


class TestSupabaseValidation:
    """The user check is an admin lookup by the token's sub; the app JWT never goes to Supabase."""

    USER = "11111111-1111-4111-8111-111111111111"

    @staticmethod
    def _client(mock_db_handler, **admin):
        mock_db = MagicMock()
        for name, value in admin.items():
            setattr(mock_db.auth.admin.get_user_by_id, name, value)
        mock_db_handler.return_value.get_client = AsyncMock(return_value=mock_db)
        return mock_db

    @pytest.mark.asyncio
    @patch('app.utils.auth_middleware.SupabaseDBHandler')
    async def test_existing_user_passes(self, mock_db_handler):
        from app.utils.auth_middleware import validate_user_with_supabase

        db = self._client(mock_db_handler, return_value=MagicMock(user=MagicMock(id=self.USER)))
        assert await validate_user_with_supabase(self.USER) is True
        db.auth.admin.get_user_by_id.assert_called_once_with(self.USER)
        db.auth.get_user.assert_not_called()

    @pytest.mark.asyncio
    @patch('app.utils.auth_middleware.SupabaseDBHandler')
    async def test_lookup_runs_off_the_event_loop(self, mock_db_handler):
        import threading
        from app.utils.auth_middleware import validate_user_with_supabase

        seen = []
        self._client(mock_db_handler, side_effect=lambda uid: seen.append(threading.current_thread()) or MagicMock())
        await validate_user_with_supabase(self.USER)
        assert seen and seen[0] is not threading.main_thread()

    @pytest.mark.asyncio
    @patch('app.utils.auth_middleware.SupabaseDBHandler')
    async def test_no_user_returned_is_401(self, mock_db_handler):
        from app.utils.auth_middleware import validate_user_with_supabase

        self._client(mock_db_handler, return_value=MagicMock(user=None))
        with pytest.raises(AuthenticationError):
            await validate_user_with_supabase(self.USER)

    @pytest.mark.asyncio
    @patch('app.utils.auth_middleware.SupabaseDBHandler')
    async def test_deleted_user_is_401(self, mock_db_handler):
        from supabase import AuthApiError
        from app.utils.auth_middleware import validate_user_with_supabase

        self._client(mock_db_handler, side_effect=AuthApiError("User not found", 404, "user_not_found"))
        with pytest.raises(AuthenticationError) as exc_info:
            await validate_user_with_supabase(self.USER)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch('app.utils.auth_middleware.SupabaseDBHandler')
    async def test_client_failure_is_401(self, mock_db_handler):
        from app.utils.auth_middleware import validate_user_with_supabase

        mock_db_handler.return_value.get_client = AsyncMock(side_effect=Exception("Database error"))
        with pytest.raises(AuthenticationError):
            await validate_user_with_supabase(self.USER)


class TestTokenCheckIndependentOfSupabaseSecret:
    """An app token signed with JWT_SECRET_KEY (not the Supabase project secret) reaches protected routes."""

    def test_protected_route_with_app_token_and_real_validator(self, client, monkeypatch):
        from app.db import base
        from app.utils.auth_middleware import jwt_handler
        from app.api.v1 import car_route

        user = "11111111-1111-4111-8111-111111111111"
        fake = MagicMock()
        fake.auth.get_user.side_effect = AssertionError("app JWT must not be sent to Supabase")
        fake.auth.admin.get_user_by_id.return_value = MagicMock(user=MagicMock(id=user))
        monkeypatch.setattr(base.SupabaseDBHandler(), "_client", fake)
        monkeypatch.setattr(car_route.user_profile_service, "get_user_cars", AsyncMock(return_value=[]))

        token = jwt_handler.create_access_token({"sub": user})
        r = client.get("/api/v1/cars", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        fake.auth.admin.get_user_by_id.assert_called_once_with(user)

    def test_deleted_user_gets_401_on_protected_route(self, client, monkeypatch):
        from supabase import AuthApiError
        from app.db import base
        from app.utils.auth_middleware import jwt_handler

        fake = MagicMock()
        fake.auth.admin.get_user_by_id.side_effect = AuthApiError("User not found", 404, "user_not_found")
        monkeypatch.setattr(base.SupabaseDBHandler(), "_client", fake)

        token = jwt_handler.create_access_token({"sub": "11111111-1111-4111-8111-111111111111"})
        r = client.get("/api/v1/cars", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401
        assert "User not found" not in r.text


class TestErrorLogsHideDetails:
    """Error logs carry only the exception class, never its message."""

    @pytest.mark.asyncio
    @patch('app.utils.auth_middleware.SupabaseDBHandler')
    async def test_supabase_validation_logs_class_only(self, mock_db_handler, caplog):
        import logging
        from app.utils.auth_middleware import validate_user_with_supabase

        mock_db = MagicMock()
        mock_db.auth.admin.get_user_by_id.side_effect = RuntimeError("leaked eyJ-token-body")
        mock_db_handler.return_value.get_client = AsyncMock(return_value=mock_db)

        with caplog.at_level(logging.DEBUG), pytest.raises(AuthenticationError):
            await validate_user_with_supabase("user-123")
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
