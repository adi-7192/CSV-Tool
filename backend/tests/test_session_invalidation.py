"""
Test session invalidation using token_version.

Verifies that when a user changes their password:
1. All existing tokens become invalid (401 on /api/auth/me)
2. New login after password change works
3. Multiple concurrent sessions are all invalidated
"""
import pytest
import sys
from pathlib import Path

# Add backend directory to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient
from main import app
from core.database import init_database, close_connection
from services.user_service import create_user, get_user_by_id
from models.user import UserCreate
from services.password_reset_service import create_password_reset_token, update_user_password
from services.auth_service import hash_password


@pytest.fixture(scope="function")
def isolated_test_db(tmp_path, monkeypatch):
    """
    Create an isolated test database for each test.
    """
    db_path = tmp_path / "test_session_invalidation.duckdb"
    
    # Patch the settings to use the test database path BEFORE any imports
    from core import config, database
    
    # Store original values
    original_db_path = config.settings.DATABASE_PATH
    original_db_path_global = database._db_path
    
    # Patch settings
    monkeypatch.setattr(config.settings, "DATABASE_PATH", str(db_path))
    
    # Reset the global database path to force re-initialization
    database._db_path = None
    
    # Close any existing connections
    try:
        close_connection()
    except:
        pass
    
    # Initialize database with test path (this will create all tables)
    init_database()
    
    yield db_path
    
    # Cleanup
    try:
        close_connection()
    except:
        pass
    
    # Restore original values
    monkeypatch.setattr(config.settings, "DATABASE_PATH", original_db_path)
    database._db_path = original_db_path_global
    
    # Clean up test database file
    if db_path.exists():
        try:
            db_path.unlink()
        except:
            pass


@pytest.fixture
def test_user(isolated_test_db):
    """
    Create a test user for session invalidation tests.
    """
    user_data = UserCreate(
        email="test_session@example.com",
        password="TestPassword123",
        role="user",
        plan="free",
        onboarded=False,
        tenant_id=None
    )
    
    user = create_user(user_data)
    return user


@pytest.fixture
def client():
    """
    Create a FastAPI test client.
    """
    return TestClient(app)


def test_password_reset_invalidates_all_sessions(isolated_test_db, test_user, client):
    """
    Test that resetting password invalidates all existing sessions.
    
    Steps:
    1. Login twice to get two tokens
    2. Verify both tokens work with /api/auth/me
    3. Reset password
    4. Verify both old tokens fail with 401
    5. Login with new password and verify new token works
    """
    # Step 1: Login twice to get two tokens
    login_response1 = client.post(
        "/api/auth/login",
        json={"email": "test_session@example.com", "password": "TestPassword123"}
    )
    assert login_response1.status_code == 200, f"Login failed: {login_response1.text}"
    token1 = login_response1.json()["access_token"]
    
    login_response2 = client.post(
        "/api/auth/login",
        json={"email": "test_session@example.com", "password": "TestPassword123"}
    )
    assert login_response2.status_code == 200, f"Login failed: {login_response2.text}"
    token2 = login_response2.json()["access_token"]
    
    # Step 2: Verify both tokens work
    response1 = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token1}"}
    )
    assert response1.status_code == 200, f"Token1 should work: {response1.text}"
    assert response1.json()["email"] == "test_session@example.com"
    
    response2 = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token2}"}
    )
    assert response2.status_code == 200, f"Token2 should work: {response2.text}"
    assert response2.json()["email"] == "test_session@example.com"
    
    # Step 3: Reset password (simulating password reset flow)
    # For testing, we'll directly update the password using update_user_password
    # This simulates what happens in the reset-password endpoint and will increment token_version
    from services.password_reset_service import update_user_password
    from services.auth_service import hash_password
    
    # Reset password directly (bypassing token flow for testing, but same effect)
    new_password_hash = hash_password("NewPassword123")
    update_user_password(test_user.id, new_password_hash)
    
    # Step 4: Verify both old tokens fail with 401
    response1_after = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token1}"}
    )
    assert response1_after.status_code == 401, f"Token1 should be invalid after password reset, got {response1_after.status_code}: {response1_after.text}"
    
    response2_after = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token2}"}
    )
    assert response2_after.status_code == 401, f"Token2 should be invalid after password reset, got {response2_after.status_code}: {response2_after.text}"
    
    # Step 5: Login with new password and verify new token works
    new_login_response = client.post(
        "/api/auth/login",
        json={"email": "test_session@example.com", "password": "NewPassword123"}
    )
    assert new_login_response.status_code == 200, f"Login with new password failed: {new_login_response.text}"
    new_token = new_login_response.json()["access_token"]
    
    new_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {new_token}"}
    )
    assert new_response.status_code == 200, f"New token should work: {new_response.text}"
    assert new_response.json()["email"] == "test_session@example.com"


def test_password_change_invalidates_all_sessions(isolated_test_db, test_user, client):
    """
    Test that changing password (via /api/auth/change-password) invalidates all existing sessions.
    
    Steps:
    1. Login to get a token
    2. Verify token works with /api/auth/me
    3. Change password using /api/auth/change-password
    4. Verify old token fails with 401
    5. Login with new password and verify new token works
    """
    # Step 1: Login to get a token
    login_response = client.post(
        "/api/auth/login",
        json={"email": "test_session@example.com", "password": "TestPassword123"}
    )
    assert login_response.status_code == 200, f"Login failed: {login_response.text}"
    token = login_response.json()["access_token"]
    
    # Step 2: Verify token works
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200, f"Token should work: {response.text}"
    assert response.json()["email"] == "test_session@example.com"
    
    # Step 3: Change password
    change_password_response = client.post(
        "/api/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "current_password": "TestPassword123",
            "new_password": "ChangedPassword123"
        }
    )
    assert change_password_response.status_code == 200, f"Password change failed: {change_password_response.text}"
    
    # Step 4: Verify old token fails with 401
    response_after = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response_after.status_code == 401, f"Token should be invalid after password change, got {response_after.status_code}: {response_after.text}"
    
    # Step 5: Login with new password and verify new token works
    new_login_response = client.post(
        "/api/auth/login",
        json={"email": "test_session@example.com", "password": "ChangedPassword123"}
    )
    assert new_login_response.status_code == 200, f"Login with new password failed: {new_login_response.text}"
    new_token = new_login_response.json()["access_token"]
    
    new_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {new_token}"}
    )
    assert new_response.status_code == 200, f"New token should work: {new_response.text}"
    assert new_response.json()["email"] == "test_session@example.com"


def test_token_version_increments_on_password_change(isolated_test_db, test_user):
    """
    Test that token_version increments when password is changed.
    
    Steps:
    1. Get initial token_version
    2. Change password
    3. Verify token_version has incremented
    """
    from services.user_service import get_user_by_id
    from services.password_reset_service import update_user_password
    from services.auth_service import hash_password
    
    # Step 1: Get initial token_version
    user = get_user_by_id(test_user.id)
    initial_token_version = user.token_version
    assert initial_token_version == 0, "New users should start with token_version=0"
    
    # Step 2: Change password
    new_password_hash = hash_password("NewPassword123")
    update_user_password(test_user.id, new_password_hash)
    
    # Step 3: Verify token_version has incremented
    user_after = get_user_by_id(test_user.id)
    assert user_after.token_version == initial_token_version + 1, \
        f"token_version should increment from {initial_token_version} to {initial_token_version + 1}, got {user_after.token_version}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

