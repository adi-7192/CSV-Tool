
import pytest
import os
from fastapi.testclient import TestClient
from main import app
from services.auth_service import create_access_token
from core.database import get_connection, init_database
from core.config import settings

# Override database path for testing
settings.DATABASE_PATH = "test_users.duckdb"

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    # Initialize DB
    if os.path.exists("test_users.duckdb"):
        os.remove("test_users.duckdb")
    
    init_database()
    
    yield
    
    # Cleanup
    try:
        if os.path.exists("test_users.duckdb"):
            os.remove("test_users.duckdb")
    except:
        pass

@pytest.fixture
def admin_token():
    # create admin user if not exists
    conn = get_connection()
    try:
        conn.execute("INSERT INTO users (id, email, password_hash, role, is_active) VALUES (999, 'admin@test.com', 'hash', 'admin', TRUE)")
    except Exception as e:
        # If user exists, update
        conn.execute("UPDATE users SET role='admin', is_active=TRUE WHERE id=999")
    

    return create_access_token(
        user_id=999,
        email="admin@test.com",
        role="admin"
    )


@pytest.fixture
def target_user_id():
    # create target user
    conn = get_connection()
    try:
        conn.execute("INSERT INTO users (id, email, password_hash, role, is_active) VALUES (1000, 'target@test.com', 'hash', 'user', TRUE)")
    except:
        conn.execute("UPDATE users SET is_active=TRUE WHERE id=1000")
    return 1000

def test_deactivate_user(admin_token, target_user_id):
    headers = {"Authorization": f"Bearer {admin_token}"}
    response = client.patch(f"/api/admin/users/{target_user_id}/deactivate", headers=headers)
    assert response.status_code == 200
    assert response.json()["success"] == True
    
    # Verify in DB
    conn = get_connection()
    status = conn.execute("SELECT is_active FROM users WHERE id = ?", [target_user_id]).fetchone()[0]
    assert status == False

def test_activate_user(admin_token, target_user_id):
    # First deactivate
    conn = get_connection()
    conn.execute("UPDATE users SET is_active=FALSE WHERE id = ?", [target_user_id])
    
    headers = {"Authorization": f"Bearer {admin_token}"}
    response = client.patch(f"/api/admin/users/{target_user_id}/activate", headers=headers)
    assert response.status_code == 200
    assert response.json()["success"] == True
    
    # Verify in DB
    status = conn.execute("SELECT is_active FROM users WHERE id = ?", [target_user_id]).fetchone()[0]
    assert status == True
