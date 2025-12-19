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



def run_migrations(conn):
    """
    Run database migrations (schema updates)
    """
    # 1. Ingestion Log Schema
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
    
    # 2. Users Schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
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
    
    # 3. User API Keys Schema
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
    
    # 4. Password Reset Tokens Schema
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
    
    # 5. System Events Schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_events (
            id BIGINT PRIMARY KEY,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            level VARCHAR NOT NULL,
            category VARCHAR,
            message VARCHAR,
            endpoint VARCHAR,
            method VARCHAR,
            status_code INTEGER,
            duration_ms INTEGER,
            tenant_id VARCHAR,
            user_id VARCHAR,
            request_id VARCHAR,
            meta VARCHAR
        )
    """)

    # 6. Apply Column Migrations (Idempotent)
    migrations = [
        # Table, Column, Definition
        ('ingestion_log', 'file_hash', 'VARCHAR'),
        ('ingestion_log', 'file_size', 'INTEGER'),
        ('ingestion_log', 'tenant_id', 'VARCHAR'),
        ('users', 'plan', "VARCHAR DEFAULT 'free'"),
        ('users', 'onboarded', "BOOLEAN DEFAULT FALSE"),
        ('users', 'is_active', "BOOLEAN DEFAULT TRUE"),
        ('users', 'last_login_at', 'TIMESTAMP'),
        ('users', 'password_changed_at', 'TIMESTAMP'),
        ('users', 'token_version', 'INTEGER NOT NULL DEFAULT 0'),
        ('sales', 'tenant_id', 'VARCHAR'), # Tenant Isolation
        ('user_api_keys', 'enabled', 'BOOLEAN DEFAULT TRUE'),
    ]
    
    for table, col, definition in migrations:
        try:
            # Check if table exists first
            tables = conn.execute("SHOW TABLES").fetchdf()
            if table in tables['name'].values:
                # Try to add column, ignore if exists
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")
                    logger.info(f"✅ Applied migration: Added '{col}' to {table}")
                except Exception:
                    pass # Column likely exists
        except Exception as e:
            logger.warning(f"Migration check failed for {table}.{col}: {e}")

    # 7. Create Indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
        "CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_hash ON password_reset_tokens(token_hash)",
        "CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_user_id ON password_reset_tokens(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_system_events_created_at ON system_events(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_system_events_level ON system_events(level)",
        "CREATE INDEX IF NOT EXISTS idx_system_events_category ON system_events(category)",
        "CREATE INDEX IF NOT EXISTS idx_system_events_endpoint ON system_events(endpoint)",
        "CREATE INDEX IF NOT EXISTS idx_system_events_tenant_id ON system_events(tenant_id)",
        "CREATE INDEX IF NOT EXISTS idx_system_events_request_id ON system_events(request_id)",
    ]
    
    for idx_sql in indexes:
        try:
            conn.execute(idx_sql)
        except Exception:
            pass


def init_database():
    """
    Initialize database - create tables if they don't exist
    
    Call this on application startup.
    """
    try:
        conn = get_connection()
        
        # Verify basic connectivity/tables
        try:
            conn.execute("SHOW TABLES")
        except Exception:
             logger.warning("Could not check tables - database might be new")
        
        run_migrations(conn)
        
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

