"""
User service - Database operations for users
"""
import logging
from typing import Optional
from datetime import datetime, timezone
import pandas as pd
from core.database import get_connection
from models.user import UserInDB, UserCreate
from services.auth_service import hash_password

logger = logging.getLogger(__name__)


def create_user(user_data: UserCreate) -> UserInDB:
    """
    Create a new user in the database.
    
    Args:
        user_data: User creation data
        
    Returns:
        Created user object
        
    Raises:
        ValueError: If email already exists
    """
    conn = get_connection()
    
    # Check if email already exists
    existing = conn.execute(
        "SELECT id FROM users WHERE email = ?",
        [user_data.email]
    ).fetchdf()
    
    if not existing.empty:
        raise ValueError(f"User with email {user_data.email} already exists")
    
    # Hash password
    password_hash = hash_password(user_data.password)
    
    # Generate user ID (simple auto-increment for DuckDB)
    max_id_result = conn.execute("SELECT COALESCE(MAX(id), 0) as max_id FROM users").fetchdf()
    new_id = int(max_id_result.iloc[0]['max_id']) + 1 if not max_id_result.empty else 1
    
    # Insert user
    try:
        from datetime import timezone as tz
        now = datetime.now(tz.utc)
    except ImportError:
        now = datetime.utcnow()
    
    # TENANT ISOLATION: Set tenant_id = user.id (as string) for strict data isolation
    # Each user's data is scoped to their own tenant_id
    tenant_id = str(new_id)
    
    conn.execute("""
        INSERT INTO users (id, email, password_hash, role, plan, onboarded, tenant_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        new_id,
        user_data.email,
        password_hash,
        user_data.role,
        user_data.plan,
        user_data.onboarded,
        tenant_id,  # Always set to user's own ID for isolation
        now
    ])
    
    logger.info(f"Created user {new_id} with email {user_data.email}, tenant_id={tenant_id}")
    
    # Return created user
    return get_user_by_id(new_id)


def get_user_by_email(email: str) -> Optional[UserInDB]:
    """
    Get user by email.
    
    Args:
        email: User email
        
    Returns:
        User object or None if not found
    """
    conn = get_connection()
    result = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        [email]
    ).fetchdf()
    
    if result.empty:
        return None
    
    row = result.iloc[0]
    return UserInDB(
        id=int(row['id']),
        email=row['email'],
        password_hash=row['password_hash'],
        role=row['role'],
        plan=row.get('plan', 'free'),
        onboarded=bool(row.get('onboarded', False)),
        tenant_id=row.get('tenant_id'),
        created_at=row['created_at'],
        token_version=int(row.get('token_version', 0))
    )


def get_user_by_id(user_id: int) -> Optional[UserInDB]:
    """
    Get user by ID.
    
    Args:
        user_id: User ID
        
    Returns:
        User object or None if not found
    """
    conn = get_connection()
    result = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        [user_id]
    ).fetchdf()
    
    if result.empty:
        return None
    
    row = result.iloc[0]
    return UserInDB(
        id=int(row['id']),
        email=row['email'],
        password_hash=row['password_hash'],
        role=row['role'],
        plan=row.get('plan', 'free'),
        onboarded=bool(row.get('onboarded', False)),
        tenant_id=row.get('tenant_id'),
        created_at=row['created_at'],
        token_version=int(row.get('token_version', 0))
    )


def get_all_users() -> list[UserInDB]:
    """
    Get all users from the database.
    
    Returns:
        List of UserInDB objects
    """
    conn = get_connection()
    result = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchdf()
    
    if result.empty:
        return []
    
    users = []
    for _, row in result.iterrows():
        users.append(UserInDB(
            id=int(row['id']),
            email=row['email'],
            password_hash=row['password_hash'],
            role=row['role'],
            plan=row.get('plan', 'free'),
            onboarded=bool(row.get('onboarded', False)),
            tenant_id=row.get('tenant_id'),
            created_at=row['created_at'],
            token_version=int(row.get('token_version', 0))
        ))
    
    return users

