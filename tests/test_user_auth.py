import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

import pytest
from fastapi import status
from main import app
from fastapi.testclient import TestClient

@pytest.mark.asyncio
async def test_register_and_login_user(monkeypatch):
    # Mock SupabaseDBHandler and its client for registration and login
    class DummyAuth:
        def sign_up(self, data):
            return type('obj', (object,), {'user': {
                'id': 'testid',
                'email': data['email'],
                'created_at': None,
                'phone': data.get('phone'),
                'user_metadata': None,
                'app_metadata': None,
                'confirmed_at': None,
                'last_sign_in_at': None,
                'role': None,
                'cars': None
            }})
        def sign_in_with_password(self, data):
            if data['email'] == 'test@example.com' and data['password'] == 'testpass':
                return type('obj', (object,), {'session': type('sess', (object,), {'access_token': 'dummy_token'})()})
            raise Exception('Invalid credentials')
        def sign_out(self, token):
            return {'message': 'Logged out successfully.'}
    class DummyClient:
        auth = DummyAuth()
    from app.db.base import SupabaseDBHandler
    # Patch the client property to return a coroutine for await compatibility
    async def dummy_client_coroutine(self):
        return DummyClient()
    monkeypatch.setattr(SupabaseDBHandler, 'client', property(dummy_client_coroutine))

    with TestClient(app) as ac:
        # Register
        response = ac.post("/api/v1/auth/register", json={
            "email": "test@example.com",
            "password": "testpass",
            "phone": "1234567890"
        })
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == "test@example.com"
        # Login
        response = ac.post("/api/v1/auth/login", data={
            "username": "test@example.com",
            "password": "testpass"
        })
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        # Logout
        response = ac.post("/api/v1/auth/logout", params={"token": "dummy_token"})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Logged out successfully."
