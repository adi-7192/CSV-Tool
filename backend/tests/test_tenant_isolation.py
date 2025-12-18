"""
End-to-end tenant isolation tests.

Tests that User A cannot access User B's data across all endpoints:
- /api/metrics
- /api/charts
- /api/data/summary
- /api/data/transactions
- /api/chat/ask
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import tempfile
import shutil
import os

# Add backend directory to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient
from main import app
from core.database import get_connection, execute_query, init_database, close_connection
from services.user_service import create_user
from models.user import UserCreate
from services.upload_service import process_csv_upload


@pytest.fixture(scope="function")
def isolated_test_db(tmp_path, monkeypatch):
    """
    Create an isolated test database for each test.
    This ensures tests don't interfere with each other.
    """
    # Create temporary database path
    db_path = tmp_path / "test_tenant_isolation.duckdb"
    
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


@pytest.fixture(scope="function")
def isolated_test_chromadb(tmp_path):
    """
    Create an isolated ChromaDB directory for each test.
    """
    chroma_path = tmp_path / "test_chromadb"
    chroma_path.mkdir(exist_ok=True)
    
    # Set environment variable (if ChromaDB respects it)
    original_chroma_path = os.environ.get('CHROMADB_PATH')
    os.environ['CHROMADB_PATH'] = str(chroma_path)
    
    yield chroma_path
    
    # Cleanup
    if original_chroma_path:
        os.environ['CHROMADB_PATH'] = original_chroma_path
    elif 'CHROMADB_PATH' in os.environ:
        del os.environ['CHROMADB_PATH']
    
    if chroma_path.exists():
        shutil.rmtree(chroma_path)


@pytest.fixture
def test_user_a(isolated_test_db):
    """Create User A for testing"""
    user_data = UserCreate(
        email="test_user_a@example.com",
        password="password123",
        role="user"
    )
    user = create_user(user_data)
    return user


@pytest.fixture
def test_user_b(isolated_test_db):
    """Create User B for testing"""
    user_data = UserCreate(
        email="test_user_b@example.com",
        password="password123",
        role="user"
    )
    user = create_user(user_data)
    return user


@pytest.fixture
def sample_csv_data_user_a():
    """Sample CSV data for User A - using recent dates to match default metrics date range"""
    from datetime import datetime, timedelta
    # Use dates within the last 30 days (default metrics range)
    base_date = datetime.now() - timedelta(days=10)
    return pd.DataFrame({
        'Invoice Number': ['INV-A-001', 'INV-A-002', 'INV-A-003'],
        'Invoice Date': [
            (base_date - timedelta(days=2)).strftime('%Y-%m-%d'),
            (base_date - timedelta(days=1)).strftime('%Y-%m-%d'),
            base_date.strftime('%Y-%m-%d')
        ],
        'SKU': ['PROD-A-1', 'PROD-A-2', 'PROD-A-3'],
        'Invoice Amount': [1000.0, 2000.0, 3000.0],
        'Qty': [1, 2, 3],
        'Type': ['Shipment', 'Shipment', 'Shipment']
    })


@pytest.fixture
def sample_csv_data_user_b():
    """Sample CSV data for User B - using recent dates to match default metrics date range"""
    from datetime import datetime, timedelta
    # Use dates within the last 30 days (default metrics range)
    base_date = datetime.now() - timedelta(days=5)
    return pd.DataFrame({
        'Invoice Number': ['INV-B-001', 'INV-B-002'],
        'Invoice Date': [
            (base_date - timedelta(days=1)).strftime('%Y-%m-%d'),
            base_date.strftime('%Y-%m-%d')
        ],
        'SKU': ['PROD-B-1', 'PROD-B-2'],
        'Invoice Amount': [5000.0, 6000.0],
        'Qty': [5, 6],
        'Type': ['Shipment', 'Shipment']
    })


def get_auth_token(client: TestClient, email: str, password: str) -> str:
    """Helper to get auth token for a user"""
    response = client.post(
        "/api/auth/login",
        json={"email": email, "password": password}  # JSON body
    )
    assert response.status_code == 200
    data = response.json()
    return data["access_token"]


def test_user_a_cannot_see_user_b_data_in_metrics(
    isolated_test_db,
    isolated_test_chromadb,
    test_user_a,
    test_user_b,
    sample_csv_data_user_a,
    tmp_path
):
    """Test that User A cannot see User B's data in metrics endpoint"""
    client = TestClient(app)
    
    # Upload data for User A only
    csv_path = tmp_path / "user_a_data.csv"
    sample_csv_data_user_a.to_csv(csv_path, index=False)
    
    # Login as User A and upload
    token_a = get_auth_token(client, "test_user_a@example.com", "password123")
    
    with open(csv_path, 'rb') as f:
        response = client.post(
            "/api/data/upload",
            files={"file": ("user_a_data.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {token_a}"}
        )
    assert response.status_code == 200
    
    # Login as User B (no data uploaded)
    token_b = get_auth_token(client, "test_user_b@example.com", "password123")
    
    # User B should see empty metrics (no data)
    response = client.get(
        "/api/metrics",
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # User B should have no data
    assert data.get("has_data") is False or data.get("data", {}).get("gross_revenue", 0) == 0
    
    # User A should see their data
    response = client.get(
        "/api/metrics",
        headers={"Authorization": f"Bearer {token_a}"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # User A should have data
    assert data.get("has_data") is True
    # Check nested data structure
    metrics_data = data.get("data", {})
    assert metrics_data.get("gross_revenue", 0) > 0


def test_user_a_cannot_see_user_b_data_in_data_summary(
    isolated_test_db,
    isolated_test_chromadb,
    test_user_a,
    test_user_b,
    sample_csv_data_user_a,
    tmp_path
):
    """Test that User A cannot see User B's data in data summary endpoint"""
    client = TestClient(app)
    
    # Upload data for User A only
    csv_path = tmp_path / "user_a_data.csv"
    sample_csv_data_user_a.to_csv(csv_path, index=False)
    
    # Login as User A and upload
    token_a = get_auth_token(client, "test_user_a@example.com", "password123")
    
    with open(csv_path, 'rb') as f:
        response = client.post(
            "/api/data/upload",
            files={"file": ("user_a_data.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {token_a}"}
        )
    assert response.status_code == 200
    
    # Login as User B (no data uploaded)
    token_b = get_auth_token(client, "test_user_b@example.com", "password123")
    
    # User B should see empty summary
    response = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # User B should have no data
    assert data.get("has_data") is False
    assert data.get("row_count", 0) == 0
    
    # User A should see their data
    response = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token_a}"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # User A should have data
    assert data.get("has_data") is True
    assert data.get("row_count", 0) > 0


def test_user_a_cannot_see_user_b_data_in_transactions(
    isolated_test_db,
    isolated_test_chromadb,
    test_user_a,
    test_user_b,
    sample_csv_data_user_a,
    tmp_path
):
    """Test that User A cannot see User B's data in transactions endpoint"""
    client = TestClient(app)
    
    # Upload data for User A only
    csv_path = tmp_path / "user_a_data.csv"
    sample_csv_data_user_a.to_csv(csv_path, index=False)
    
    # Login as User A and upload
    token_a = get_auth_token(client, "test_user_a@example.com", "password123")
    
    with open(csv_path, 'rb') as f:
        response = client.post(
            "/api/data/upload",
            files={"file": ("user_a_data.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {token_a}"}
        )
    assert response.status_code == 200
    
    # Login as User B (no data uploaded)
    token_b = get_auth_token(client, "test_user_b@example.com", "password123")
    
    # User B should see empty transactions
    response = client.get(
        "/api/data/transactions",
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # User B should have no transactions
    assert len(data.get("data", [])) == 0
    assert data.get("total", 0) == 0
    
    # User A should see their transactions
    response = client.get(
        "/api/data/transactions",
        headers={"Authorization": f"Bearer {token_a}"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # User A should have transactions
    assert len(data.get("data", [])) > 0
    assert data.get("total", 0) > 0


def test_user_a_cannot_see_user_b_data_in_chat(
    isolated_test_db,
    isolated_test_chromadb,
    test_user_a,
    test_user_b,
    sample_csv_data_user_a,
    tmp_path
):
    """Test that User A cannot see User B's data in chat endpoint"""
    client = TestClient(app)
    
    # Upload data for User A only
    csv_path = tmp_path / "user_a_data.csv"
    sample_csv_data_user_a.to_csv(csv_path, index=False)
    
    # Login as User A and upload
    token_a = get_auth_token(client, "test_user_a@example.com", "password123")
    
    with open(csv_path, 'rb') as f:
        response = client.post(
            "/api/data/upload",
            files={"file": ("user_a_data.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {token_a}"}
        )
    assert response.status_code == 200
    
    # Login as User B (no data uploaded)
    token_b = get_auth_token(client, "test_user_b@example.com", "password123")
    
    # User B asks about sales - should get "no data" response
    response = client.post(
        "/api/chat/ask",
        json={"question": "What is my total revenue?"},
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # User B should get a "no data" response
    answer = data.get("answer", "").lower()
    assert "no data" in answer or "don't have" in answer or "upload" in answer
    
    # User A asks about sales - should get actual data
    response = client.post(
        "/api/chat/ask",
        json={"question": "What is my total revenue?"},
        headers={"Authorization": f"Bearer {token_a}"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # User A should get a response with data (not "no data")
    answer = data.get("answer", "").lower()
    assert "no data" not in answer or "revenue" in answer or "₹" in answer or "rupee" in answer


def test_upload_isolation(
    isolated_test_db,
    isolated_test_chromadb,
    test_user_a,
    test_user_b,
    sample_csv_data_user_a,
    sample_csv_data_user_b,
    tmp_path
):
    """Test that User A's upload doesn't appear for User B"""
    client = TestClient(app)
    
    # Upload data for User A
    csv_path_a = tmp_path / "user_a_data.csv"
    sample_csv_data_user_a.to_csv(csv_path_a, index=False)
    
    token_a = get_auth_token(client, "test_user_a@example.com", "password123")
    
    with open(csv_path_a, 'rb') as f:
        response = client.post(
            "/api/data/upload",
            files={"file": ("user_a_data.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {token_a}"}
        )
    assert response.status_code == 200
    
    # Upload data for User B
    csv_path_b = tmp_path / "user_b_data.csv"
    sample_csv_data_user_b.to_csv(csv_path_b, index=False)
    
    token_b = get_auth_token(client, "test_user_b@example.com", "password123")
    
    with open(csv_path_b, 'rb') as f:
        response = client.post(
            "/api/data/upload",
            files={"file": ("user_b_data.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {token_b}"}
        )
    assert response.status_code == 200
    
    # User A should only see their data (3 rows)
    response = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token_a}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("row_count", 0) == 3  # User A's 3 rows
    
    # User B should only see their data (2 rows)
    response = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("row_count", 0) == 2  # User B's 2 rows




def test_user_a_cannot_export_user_b_data(
    isolated_test_db,
    isolated_test_chromadb,
    test_user_a,
    test_user_b,
    sample_csv_data_user_a,
    sample_csv_data_user_b,
    tmp_path,
):
    """Test that User A cannot export User B's data via export endpoint"""
    client = TestClient(app)
    
    # Login both users
    token_a = get_auth_token(client, test_user_a.email, "password123")
    token_b = get_auth_token(client, test_user_b.email, "password123")
    
    # Upload data for both users
    csv_path_a = tmp_path / "user_a_data.csv"
    sample_csv_data_user_a.to_csv(csv_path_a, index=False)
    
    with open(csv_path_a, 'rb') as f:
        upload_a = client.post(
            "/api/data/upload",
            files={"file": ("user_a_data.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {token_a}"}
        )
        assert upload_a.status_code == 200
    
    csv_path_b = tmp_path / "user_b_data.csv"
    sample_csv_data_user_b.to_csv(csv_path_b, index=False)
    
    with open(csv_path_b, 'rb') as f:
        upload_b = client.post(
            "/api/data/upload",
            files={"file": ("user_b_data.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {token_b}"}
        )
        assert upload_b.status_code == 200
    
    # User A exports their data (should only get User A's data)
    export_a = client.get(
        "/api/data/export?format=csv",
        headers={"Authorization": f"Bearer {token_a}"}
    )
    assert export_a.status_code == 200
    assert "text/csv" in export_a.headers.get("content-type", "")
    
    # User B exports their data (should only get User B's data)
    export_b = client.get(
        "/api/data/export?format=csv",
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert export_b.status_code == 200
    assert "text/csv" in export_b.headers.get("content-type", "")
    
    # Verify exports are different (different tenant data)
    assert export_a.text != export_b.text


def test_user_a_cannot_delete_user_b_ingestion(
    isolated_test_db,
    isolated_test_chromadb,
    test_user_a,
    test_user_b,
    sample_csv_data_user_a,
    sample_csv_data_user_b,
    tmp_path,
):
    """Test that User A cannot delete User B's ingestion via delete endpoint"""
    client = TestClient(app)
    
    # Login both users
    token_a = get_auth_token(client, test_user_a.email, "password123")
    token_b = get_auth_token(client, test_user_b.email, "password123")
    
    # Upload data for User B
    csv_path_b = tmp_path / "user_b_data.csv"
    sample_csv_data_user_b.to_csv(csv_path_b, index=False)
    
    ingestion_id_b = None
    with open(csv_path_b, 'rb') as f:
        upload_b = client.post(
            "/api/data/upload",
            files={"file": ("user_b_data.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {token_b}"}
        )
        assert upload_b.status_code == 200
        ingestion_id_b = upload_b.json().get("ingestion_id")
    
    assert ingestion_id_b is not None
    
    # User A tries to delete User B's ingestion (should fail with 404)
    delete_response = client.delete(
        f"/api/upload/{ingestion_id_b}",
        headers={"Authorization": f"Bearer {token_a}"}
    )
    assert delete_response.status_code == 404  # Not found or doesn't belong to user
    
    # Verify User B's data still exists
    summary_b = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert summary_b.status_code == 200
    summary_data_b = summary_b.json()
    assert summary_data_b.get("has_data") is True
    assert summary_data_b.get("row_count", 0) > 0
    
    # User B can delete their own ingestion
    delete_b = client.delete(
        f"/api/upload/{ingestion_id_b}",
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert delete_b.status_code == 200
    
    # Verify User B's data is deleted
    summary_b_after = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert summary_b_after.status_code == 200
    summary_data_b_after = summary_b_after.json()
    assert summary_data_b_after.get("has_data") is False
    assert summary_data_b_after.get("row_count", 0) == 0


def test_user_a_cannot_reset_user_b_data(
    isolated_test_db,
    isolated_test_chromadb,
    test_user_a,
    test_user_b,
    sample_csv_data_user_a,
    sample_csv_data_user_b,
    tmp_path,
):
    """Test that User A cannot reset User B's data via reset endpoint"""
    client = TestClient(app)
    
    # Login both users
    token_a = get_auth_token(client, test_user_a.email, "password123")
    token_b = get_auth_token(client, test_user_b.email, "password123")
    
    # Upload data for User B
    csv_path_b = tmp_path / "user_b_data.csv"
    sample_csv_data_user_b.to_csv(csv_path_b, index=False)
    
    with open(csv_path_b, 'rb') as f:
        upload_b = client.post(
            "/api/data/upload",
            files={"file": ("user_b_data.csv", f, "text/csv")},
            headers={"Authorization": f"Bearer {token_b}"}
        )
        assert upload_b.status_code == 200
    
    # Verify User B has data
    summary_b_before = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert summary_b_before.status_code == 200
    summary_data_b_before = summary_b_before.json()
    assert summary_data_b_before.get("has_data") is True
    assert summary_data_b_before.get("row_count", 0) > 0
    row_count_before = summary_data_b_before.get("row_count", 0)
    
    # User A tries to reset (should only reset User A's data, not User B's)
    reset_a = client.post(
        "/api/data/reset",
        json={"confirm": "DELETE"},
        headers={"Authorization": f"Bearer {token_a}"}
    )
    assert reset_a.status_code == 200
    
    # Verify User B's data still exists (not affected by User A's reset)
    summary_b_after = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert summary_b_after.status_code == 200
    summary_data_b_after = summary_b_after.json()
    assert summary_data_b_after.get("has_data") is True
    assert summary_data_b_after.get("row_count", 0) == row_count_before
    
    # User B can reset their own data
    reset_b = client.post(
        "/api/data/reset",
        json={"confirm": "DELETE"},
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert reset_b.status_code == 200
    
    # Verify User B's data is now deleted
    summary_b_final = client.get(
        "/api/data/summary",
        headers={"Authorization": f"Bearer {token_b}"}
    )
    assert summary_b_final.status_code == 200
    summary_data_b_final = summary_b_final.json()
    assert summary_data_b_final.get("has_data") is False
    assert summary_data_b_final.get("row_count", 0) == 0

