#!/usr/bin/env python3
"""
Fix is_active column in users table

This script ensures the is_active column exists in the users table.
Run this if you get "Referenced update column is_active not found in table" errors.
"""
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from core.database import get_connection, table_exists, init_database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_is_active_column():
    """Add is_active column to users table if it doesn't exist"""
    try:
        # Initialize database (ensures tables exist)
        init_database()
        
        if not table_exists('users'):
            logger.error("❌ Users table does not exist. Run init_database() first.")
            return False
        
        conn = get_connection()
        
        # Check if column exists by trying to describe the table
        try:
            # Try to query the column - if it fails, it doesn't exist
            result = conn.execute("SELECT is_active FROM users LIMIT 1").fetchdf()
            logger.info("✅ is_active column already exists in users table")
            return True
        except Exception as e:
            # Column doesn't exist, add it using a workaround
            logger.info("⚠️  is_active column not found. Adding it...")
            try:
                # Method 1: Try direct ALTER TABLE first
                try:
                    conn.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE")
                    logger.info("✅ Successfully added 'is_active' column to users table")
                    
                    # Update all existing users to be active by default
                    conn.execute("UPDATE users SET is_active = TRUE WHERE is_active IS NULL")
                    logger.info("✅ Set all existing users to active (is_active = TRUE)")
                    return True
                except Exception as alter_error:
                    # If ALTER fails due to dependencies, use table recreation method
                    logger.warning(f"ALTER TABLE failed: {alter_error}")
                    logger.info("🔄 Trying alternative method: recreating table with new column...")
                    
                    # Get all existing data
                    existing_users = conn.execute("SELECT * FROM users").fetchdf()
                    logger.info(f"📊 Found {len(existing_users)} existing users to migrate")
                    
                    # Get the current schema
                    schema_info = conn.execute("DESCRIBE users").fetchdf()
                    existing_columns = schema_info['column_name'].tolist()
                    logger.info(f"📋 Current columns: {existing_columns}")
                    
                    # Create new table with is_active column
                    conn.execute("""
                        CREATE TABLE users_new (
                            id INTEGER PRIMARY KEY,
                            email VARCHAR NOT NULL UNIQUE,
                            password_hash VARCHAR NOT NULL,
                            role VARCHAR NOT NULL DEFAULT 'user',
                            plan VARCHAR NOT NULL DEFAULT 'free',
                            onboarded BOOLEAN NOT NULL DEFAULT FALSE,
                            tenant_id VARCHAR,
                            is_active BOOLEAN NOT NULL DEFAULT TRUE,
                            last_login_at TIMESTAMP,
                            password_changed_at TIMESTAMP,
                            token_version INTEGER NOT NULL DEFAULT 0,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Copy data from old table to new table
                    # Build INSERT statement with all columns, adding is_active = TRUE
                    if len(existing_users) > 0:
                        # Get column names that exist in both tables
                        common_cols = [col for col in existing_columns if col != 'is_active']
                        insert_cols_list = common_cols + ['is_active']
                        select_cols_list = [f'"{col}"' for col in common_cols] + ['TRUE']
                        
                        insert_cols = ', '.join([f'"{col}"' for col in insert_cols_list])
                        select_cols = ', '.join(select_cols_list)
                        
                        conn.execute(f"""
                            INSERT INTO users_new ({insert_cols})
                            SELECT {select_cols}
                            FROM users
                        """)
                        logger.info(f"✅ Migrated {len(existing_users)} users to new table")
                    else:
                        logger.info("ℹ️  No existing users to migrate")
                    
                    # Drop old table and rename new one
                    conn.execute("DROP TABLE users")
                    conn.execute("ALTER TABLE users_new RENAME TO users")
                    
                    # Recreate index on email
                    try:
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
                    except:
                        pass
                    
                    logger.info("✅ Successfully recreated users table with is_active column")
                    logger.info("✅ All existing users set to active (is_active = TRUE)")
                    return True
                    
            except Exception as e2:
                logger.error(f"❌ Failed to add is_active column: {e2}", exc_info=True)
                return False
        
    except Exception as e:
        logger.error(f"❌ Error fixing is_active column: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("🔧 Fixing is_active column in users table...")
    success = fix_is_active_column()
    
    if success:
        logger.info("✅ Done! The is_active column should now be available.")
        sys.exit(0)
    else:
        logger.error("❌ Failed to fix is_active column. Check the error messages above.")
        sys.exit(1)

