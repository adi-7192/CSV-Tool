"""
Database operations - DuckDB connection and query management

Extracted and refactored from legacy/db_manager.py

IMPORTANT: DuckDB connections are NOT thread-safe. We use thread-local storage
to ensure each thread gets its own connection, preventing segmentation faults.
"""
import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import threading
from core.config import settings

logger = logging.getLogger(__name__)

# Thread-local storage for connections (thread-safe)
_thread_local = threading.local()
_db_path: Optional[Path] = None


def _get_db_path() -> Path:
    """Get the database path, resolving relative paths"""
    global _db_path
    if _db_path is None:
        db_path = Path(settings.DATABASE_PATH)
        if not db_path.is_absolute():
            # If relative (starts with ../), resolve from backend/ directory
            if str(db_path).startswith("../"):
                relative_path = str(db_path)[3:]  # Remove "../"
                db_path = Path(__file__).parent.parent.parent / relative_path
            else:
                db_path = Path(__file__).parent.parent.parent / db_path
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _db_path = db_path
    
    return _db_path


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Get or create DuckDB connection (thread-local)
    
    DuckDB connections are NOT thread-safe. Each thread gets its own connection
    to prevent segmentation faults and data corruption.
    
    Returns:
        duckdb.DuckDBPyConnection: DuckDB connection instance for current thread
    """
    # Check if this thread already has a connection
    if not hasattr(_thread_local, 'connection') or _thread_local.connection is None:
        db_path = _get_db_path()
        
        try:
            _thread_local.connection = duckdb.connect(str(db_path))
            logger.debug(f"✅ DuckDB connection established for thread {threading.current_thread().name}: {db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create DuckDB connection: {e}")
            raise
    
    # Verify connection is still valid
    try:
        _thread_local.connection.execute("SELECT 1")
    except Exception as e:
        logger.warning(f"Connection invalid, recreating: {e}")
        try:
            _thread_local.connection.close()
        except:
            pass
        db_path = _get_db_path()
        _thread_local.connection = duckdb.connect(str(db_path))
        logger.info(f"✅ Recreated DuckDB connection for thread {threading.current_thread().name}")
    
    return _thread_local.connection


def init_database():
    """
    Initialize database - create tables if they don't exist
    
    Call this on application startup.
    """
    try:
        conn = get_connection()
        
        # Create sales table if it doesn't exist (it should exist from Phase 1)
        # Just verify it exists
        try:
            tables = conn.execute("SHOW TABLES").fetchdf()
            if 'sales' not in tables['name'].values:
                logger.warning("Sales table doesn't exist - will be created on first upload")
            else:
                logger.info("✅ Sales table found - database ready")
        except Exception:
            logger.warning("Could not check tables - will be created on first upload")
        
        # Create ingestion_log table if it doesn't exist (from Phase 1)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                ingestion_id VARCHAR PRIMARY KEY,
                filename VARCHAR NOT NULL,
                uploaded_at TIMESTAMP NOT NULL,
                rows_raw INTEGER,
                rows_cleaned INTEGER,
                rows_inserted INTEGER,
                date_range_start DATE,
                date_range_end DATE,
                validation_status VARCHAR,
                validation_issues JSON,
                processing_time_seconds FLOAT,
                file_hash VARCHAR,
                file_size INTEGER,
                tenant_id VARCHAR
            )
        """)
        
        # Add file_hash, file_size, and tenant_id columns if they don't exist (migration)
        try:
            conn.execute("ALTER TABLE ingestion_log ADD COLUMN file_hash VARCHAR")
            logger.info("✅ Added 'file_hash' column to ingestion_log table")
        except Exception:
            pass
        
        try:
            conn.execute("ALTER TABLE ingestion_log ADD COLUMN file_size INTEGER")
            logger.info("✅ Added 'file_size' column to ingestion_log table")
        except Exception:
            pass
        
        try:
            conn.execute("ALTER TABLE ingestion_log ADD COLUMN tenant_id VARCHAR")
            logger.info("✅ Added 'tenant_id' column to ingestion_log table")
        except Exception:
            pass
        
        # Create users table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                email VARCHAR NOT NULL UNIQUE,
                password_hash VARCHAR NOT NULL,
                role VARCHAR NOT NULL DEFAULT 'user',
                plan VARCHAR NOT NULL DEFAULT 'free',
                onboarded BOOLEAN NOT NULL DEFAULT FALSE,
                tenant_id VARCHAR,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add plan and onboarded columns if they don't exist (migration for existing databases)
        try:
            conn.execute("ALTER TABLE users ADD COLUMN plan VARCHAR DEFAULT 'free'")
            logger.info("✅ Added 'plan' column to users table")
        except Exception:
            # Column already exists, ignore
            pass
        
        try:
            conn.execute("ALTER TABLE users ADD COLUMN onboarded BOOLEAN DEFAULT FALSE")
            logger.info("✅ Added 'onboarded' column to users table")
        except Exception:
            # Column already exists, ignore
            pass
        
        # Create index on email for faster lookups
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        except Exception:
            # Index might already exist
            pass
        
        # Create user_api_keys table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_api_keys (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                provider VARCHAR NOT NULL,
                encrypted_key VARCHAR NOT NULL,
                enabled BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, provider)
            )
        """)
        
        # TENANT ISOLATION: Add tenant_id column to sales table if it doesn't exist
        # This enables strict data isolation between users
        if table_exists('sales'):
            try:
                conn.execute("ALTER TABLE sales ADD COLUMN tenant_id VARCHAR")
                logger.info("✅ Added 'tenant_id' column to sales table for tenant isolation")
            except Exception:
                # Column already exists, ignore
                pass
        
        # Add enabled column if it doesn't exist (migration for existing databases)
        try:
            conn.execute("ALTER TABLE user_api_keys ADD COLUMN enabled BOOLEAN DEFAULT TRUE")
            logger.info("✅ Added 'enabled' column to user_api_keys table")
        except Exception:
            # Column already exists, ignore
            pass
        
        # Create password_reset_tokens table for secure password reset flow
        conn.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id VARCHAR PRIMARY KEY,
                user_id INTEGER NOT NULL,
                token_hash VARCHAR NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                request_ip VARCHAR,
                user_agent VARCHAR
            )
        """)
        
        # Create indexes for efficient token lookup
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_hash ON password_reset_tokens(token_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_user_id ON password_reset_tokens(user_id)")
        except Exception:
            # Indexes might already exist
            pass
        
        # Add password_changed_at column to users table for session invalidation
        try:
            conn.execute("ALTER TABLE users ADD COLUMN password_changed_at TIMESTAMP")
            logger.info("✅ Added 'password_changed_at' column to users table")
        except Exception:
            # Column already exists, ignore
            pass
        
        # Add token_version column to users table for session invalidation
        # When password changes, token_version is incremented, invalidating all existing JWTs
        try:
            conn.execute("ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0")
            logger.info("✅ Added 'token_version' column to users table")
        except Exception:
            # Column already exists, ignore
            pass
        
        logger.info("✅ Database initialization complete")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


def execute_query(sql: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Execute SQL query and return results as DataFrame
    
    Args:
        sql: SQL query string
        params: Optional parameters for query (prevents SQL injection)
        
    Returns:
        DataFrame with query results
        
    Raises:
        Exception: If query execution fails
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            conn = get_connection()
            
            if params:
                # DuckDB parameterized queries (if supported)
                result = conn.execute(sql, list(params.values())).fetchdf()
            else:
                result = conn.execute(sql).fetchdf()
            
            logger.debug(f"Query returned {len(result)} rows")
            return result
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            
            # Check if it's a connection issue
            if "connection" in error_msg.lower() or "closed" in error_msg.lower() or retry_count < max_retries:
                logger.warning(f"Query execution failed (attempt {retry_count}/{max_retries}): {error_msg}")
                # Clear the connection for this thread to force recreation
                if hasattr(_thread_local, 'connection'):
                    try:
                        _thread_local.connection.close()
                    except:
                        pass
                    _thread_local.connection = None
                
                if retry_count < max_retries:
                    continue  # Retry with new connection
            
            logger.error(f"Query execution failed after {retry_count} attempts: {error_msg}")
            logger.error(f"SQL: {sql[:200]}...")  # Log first 200 chars to avoid huge logs
            raise


def table_exists(table_name: str) -> bool:
    """
    Check if table exists in database
    
    Uses parameterized query to prevent SQL injection.
    """
    try:
        conn = get_connection()
        # Use parameterized query for safety
        result = conn.execute("""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_name = ?
        """, [table_name]).fetchone()
        return result[0] > 0 if result else False
    except Exception as e:
        logger.debug(f"Error checking if table exists: {e}")
        return False


def get_row_count(table_name: str = 'sales') -> int:
    """Get total row count in table"""
    if not table_exists(table_name):
        return 0
    
    try:
        result = execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
        return int(result['count'].iloc[0]) if not result.empty else 0
    except Exception:
        return 0


def close_connection():
    """
    Close database connection for current thread (call on app shutdown)
    
    Note: This only closes the connection for the current thread.
    Other threads will close their connections when they finish.
    """
    if hasattr(_thread_local, 'connection') and _thread_local.connection is not None:
        try:
            _thread_local.connection.close()
            _thread_local.connection = None
            logger.debug(f"✅ Database connection closed for thread {threading.current_thread().name}")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

