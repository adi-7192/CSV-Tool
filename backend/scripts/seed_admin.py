"""
Admin Seeding Script (Development Only)

This script promotes a specific email address to admin role for development purposes.
It should NOT be exposed as a public API endpoint or run in production without proper safeguards.

Usage:
    cd backend
    python -m scripts.seed_admin

Behavior:
    - Checks if user with email "adityadav7192@gmail.com" exists
    - If NOT: Creates user with role="admin" and default password
    - If YES: Updates existing user's role to "admin" (does NOT change password)
    
Security Notes:
    - This is a one-time development script
    - The default password should be changed immediately after first login
    - In production, use proper admin promotion workflows with audit trails
"""
import sys
from pathlib import Path

# Add backend directory to path so we can import modules
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from core.database import get_connection, init_database
from services.user_service import get_user_by_email, get_user_by_id
from datetime import datetime, timezone
import logging
import bcrypt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Admin email and default password
ADMIN_EMAIL = "adityadav7192@gmail.com"
DEFAULT_PASSWORD = "ChangeThisPassword123!"


def seed_admin():
    """
    Seed or update admin user.
    
    Creates admin user if doesn't exist, or updates existing user to admin role.
    """
    try:
        # Initialize database (creates tables if needed)
        init_database()
        logger.info("✅ Database initialized")
        
        conn = get_connection()
        
        # Check if user exists
        existing_user = get_user_by_email(ADMIN_EMAIL)
        
        if existing_user:
            # User exists - update role to admin
            logger.info(f"User found: {ADMIN_EMAIL} (ID: {existing_user.id})")
            
            # Check if already admin
            if existing_user.role == "admin":
                logger.info(f"✅ User {ADMIN_EMAIL} is already an admin. No changes needed.")
                return
            
            # Update role to admin
            conn.execute("""
                UPDATE users 
                SET role = ? 
                WHERE email = ?
            """, ["admin", ADMIN_EMAIL])
            
            logger.info(f"✅ Updated user {ADMIN_EMAIL} to admin role")
            logger.info(f"   User ID: {existing_user.id}")
            logger.info(f"   Previous role: {existing_user.role}")
            logger.info(f"   New role: admin")
            
        else:
            # User doesn't exist - create admin user
            logger.info(f"User not found: {ADMIN_EMAIL}")
            logger.info("Creating new admin user...")
            
            # Generate user ID
            max_id_result = conn.execute("SELECT COALESCE(MAX(id), 0) as max_id FROM users").fetchdf()
            new_id = int(max_id_result.iloc[0]['max_id']) + 1 if not max_id_result.empty else 1
            
            # Hash password using bcrypt directly (workaround for passlib/bcrypt version issue)
            # bcrypt has a 72-byte limit, but our password is well within that
            password_bytes = DEFAULT_PASSWORD.encode('utf-8')
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(password_bytes, salt).decode('utf-8')
            
            # Insert admin user
            now = datetime.now(timezone.utc)
            conn.execute("""
                INSERT INTO users (id, email, password_hash, role, tenant_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                new_id,
                ADMIN_EMAIL,
                password_hash,
                "admin",
                None,  # tenant_id
                now
            ])
            
            logger.info(f"✅ Created admin user: {ADMIN_EMAIL}")
            logger.info(f"   User ID: {new_id}")
            logger.info(f"   Email: {ADMIN_EMAIL}")
            logger.info(f"   Role: admin")
            logger.info(f"   Default password: {DEFAULT_PASSWORD}")
            logger.warning("⚠️  IMPORTANT: Change the default password after first login!")
        
        logger.info("\n" + "="*60)
        logger.info("Admin seeding completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Error seeding admin: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    seed_admin()

