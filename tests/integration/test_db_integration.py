import pytest
from app.db.base import SupabaseDBHandler

@pytest.mark.asyncio
async def test_create_and_get_user():
    db = SupabaseDBHandler()
    # Replace with actual user creation logic and test DB config
    user_data = {"email": "test@example.com", "name": "Test User"}
    user = await db.create_user(user_data)
    assert user["email"] == user_data["email"]
    fetched = await db.get_user_by_email(user_data["email"])
    assert fetched is not None
    # Clean up if needed
    await db.delete_user(user["id"])
