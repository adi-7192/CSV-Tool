"""
Database operations - DuckDB connection and query management

Extracted and refactored from legacy/db_manager.py
"""
import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from core.config import settings

logger = logging.getLogger(__name__)

# Global connection (singleton pattern)
_connection: Optional[duckdb.DuckDBPyConnection] = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Get or create DuckDB connection (singleton)
    
    This ensures we reuse the same connection across requests.
    
    Returns:
        duckdb.DuckDBPyConnection: DuckDB connection instance
    """
    global _connection
    
    if _connection is None:
        db_path = Path(settings.DATABASE_PATH)
        if not db_path.is_absolute():
            # If relative (starts with ../), resolve from backend/ directory
            if str(db_path).startswith("../"):
                relative_path = str(db_path)[3:]  # Remove "../"
                db_path = Path(__file__).parent.parent.parent / relative_path
            else:
                db_path = Path(__file__).parent.parent.parent / db_path
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        _connection = duckdb.connect(str(db_path))
        logger.info(f"✅ DuckDB connection established: {db_path}")
    
    return _connection


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
                processing_time_seconds FLOAT
            )
        """)
        
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
        
        # Add enabled column if it doesn't exist (migration for existing databases)
        try:
            conn.execute("ALTER TABLE user_api_keys ADD COLUMN enabled BOOLEAN DEFAULT TRUE")
            logger.info("✅ Added 'enabled' column to user_api_keys table")
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
    """
    conn = get_connection()
    
    try:
        if params:
            # DuckDB parameterized queries (if supported)
            result = conn.execute(sql, list(params.values())).fetchdf()
        else:
            result = conn.execute(sql).fetchdf()
        
        logger.debug(f"Query returned {len(result)} rows")
        return result
        
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        logger.error(f"SQL: {sql}")
        raise


def table_exists(table_name: str) -> bool:
    """Check if table exists in database"""
    try:
        conn = get_connection()
        result = conn.execute(f"""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        """).fetchone()
        return result[0] > 0 if result else False
    except Exception:
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
    """Close database connection (call on app shutdown)"""
    global _connection
    if _connection:
        _connection.close()
        _connection = None
        logger.info("✅ Database connection closed")

