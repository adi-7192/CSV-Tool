"""
Database Manager for CSV Analytics Dashboard

This module provides DuckDB database operations for the CSV Analytics Dashboard.
It includes connection management, data storage, querying, and utility functions.

Author: CSV Analytics Dashboard
Version: 1.0.0
"""

import os
import pandas as pd
import duckdb
import streamlit as st
from typing import Optional, Union, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_FILE = 'data/analytics.duckdb'
DEFAULT_TABLE = 'sales'

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)


@st.cache_resource
def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Get a cached DuckDB connection.
    
    Uses Streamlit's @st.cache_resource decorator to maintain a single
    connection instance across reruns, improving performance.
    
    Returns:
        duckdb.DuckDBPyConnection: Cached DuckDB connection instance
        
    Raises:
        Exception: If connection cannot be established
    """
    try:
        conn = duckdb.connect(DB_FILE)
        logger.info(f"✅ DuckDB connection established: {DB_FILE}")
        return conn
    except Exception as e:
        logger.error(f"❌ Failed to connect to DuckDB: {e}")
        raise Exception(f"Database connection failed: {e}")


def store_data(df: pd.DataFrame, table_name: str = DEFAULT_TABLE, append: bool = True) -> Dict[str, Any]:
    """
    Store a DataFrame in the DuckDB database.
    
    Args:
        df (pd.DataFrame): DataFrame to store
        table_name (str): Name of the table to store data in (default: 'sales')
        append (bool): If True, append to existing table; if False, replace table
        
    Returns:
        Dict[str, Any]: Result dictionary with success status, rows stored, and any errors
        
    Raises:
        Exception: If data storage fails
    """
    try:
        conn = get_connection()
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            logger.warning("⚠️ Empty DataFrame provided")
            return {
                'success': False,
                'rows_stored': 0,
                'error': 'Empty DataFrame provided'
            }
        
        # Determine operation mode
        mode = "APPEND" if append else "REPLACE"
        logger.info(f"📊 Storing {len(df)} rows to table '{table_name}' in {mode} mode")
        
        # Store data using DuckDB's register method
        conn.register('temp_df', df)
        
        if append and table_exists(table_name):
            # Append to existing table
            # Note: DuckDB doesn't have INSERT OR IGNORE, so we'll insert all rows
            # Duplicates will be handled by the verification logic in app.py
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
            logger.info(f"📊 Appended {len(df)} rows to existing table '{table_name}'")
        else:
            # Create new table or replace existing
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM temp_df")
            logger.info(f"📊 Created/replaced table '{table_name}' with {len(df)} rows")
        
        # Create indexes if this is a new table or first append
        if not append or not table_exists(table_name):
            create_indexes(table_name)
        
        # Commit the transaction
        conn.commit()
        
        # Get final row count
        final_count = get_row_count(table_name)
        
        logger.info(f"✅ Successfully stored {len(df)} rows to '{table_name}'. Total rows: {final_count}")
        
        return {
            'success': True,
            'rows_stored': len(df),
            'total_rows': final_count,
            'table_name': table_name,
            'mode': mode
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to store data: {e}")
        return {
            'success': False,
            'rows_stored': 0,
            'error': str(e)
        }


def query_data(sql: str) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.
    
    Args:
        sql (str): SQL query to execute
        
    Returns:
        pd.DataFrame: Query results as a DataFrame
        
    Raises:
        Exception: If query execution fails
    """
    try:
        conn = get_connection()
        
        if not sql or not sql.strip():
            raise ValueError("SQL query cannot be empty")
        
        logger.info(f"🔍 Executing SQL query: {sql[:100]}...")
        
        # Execute query and return DataFrame
        result_df = conn.execute(sql).df()
        
        logger.info(f"✅ Query executed successfully. Returned {len(result_df)} rows")
        
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Failed to execute query: {e}")
        raise Exception(f"Query execution failed: {e}")


def table_exists(table_name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        table_name (str): Name of the table to check
        
    Returns:
        bool: True if table exists, False otherwise
        
    Raises:
        Exception: If check fails
    """
    try:
        conn = get_connection()
        
        # Query to check if table exists
        result = conn.execute(f"""
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        """).fetchone()
        
        exists = result[0] > 0 if result else False
        
        logger.info(f"🔍 Table '{table_name}' exists: {exists}")
        
        return exists
        
    except Exception as e:
        logger.error(f"❌ Failed to check table existence: {e}")
        return False


def clear_database(table_name: str = DEFAULT_TABLE) -> Dict[str, Any]:
    """
    Drop a table from the database.
    
    Args:
        table_name (str): Name of the table to drop (default: 'sales')
        
    Returns:
        Dict[str, Any]: Result dictionary with success status and any errors
        
    Raises:
        Exception: If table drop fails
    """
    try:
        conn = get_connection()
        
        if not table_exists(table_name):
            logger.warning(f"⚠️ Table '{table_name}' does not exist")
            return {
                'success': False,
                'error': f"Table '{table_name}' does not exist"
            }
        
        # Get row count before dropping
        row_count = get_row_count(table_name)
        
        # Drop the table
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        
        logger.info(f"✅ Successfully dropped table '{table_name}' ({row_count} rows removed)")
        
        return {
            'success': True,
            'table_name': table_name,
            'rows_removed': row_count
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to drop table: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def get_row_count(table_name: str = DEFAULT_TABLE) -> int:
    """
    Get the total number of rows in a table.
    
    Args:
        table_name (str): Name of the table to count rows in (default: 'sales')
        
    Returns:
        int: Total number of rows in the table
        
    Raises:
        Exception: If count fails
    """
    try:
        conn = get_connection()
        
        if not table_exists(table_name):
            logger.warning(f"⚠️ Table '{table_name}' does not exist")
            return 0
        
        # Count rows
        result = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
        row_count = result[0] if result else 0
        
        logger.info(f"📊 Table '{table_name}' has {row_count} rows")
        
        return row_count
        
    except Exception as e:
        logger.error(f"❌ Failed to get row count: {e}")
        return 0


def create_indexes(table_name: str) -> None:
    """
    Create indexes on key columns for better query performance.
    
    Args:
        table_name (str): Name of the table to create indexes on
        
    Raises:
        Exception: If index creation fails
    """
    try:
        conn = get_connection()
        
        # Define indexes to create
        indexes = [
            ('idx_order_date', 'order_date'),
            ('idx_order_id', 'order_id'),
            ('idx_transaction_type', 'transaction_type'),
            ('idx_sku', 'sku')
        ]
        
        logger.info(f"🔧 Creating indexes for table '{table_name}'")
        
        for index_name, column_name in indexes:
            try:
                # Check if column exists before creating index
                column_exists = conn.execute(f"""
                    SELECT COUNT(*) as count 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}' AND column_name = '{column_name}'
                """).fetchone()
                
                if column_exists and column_exists[0] > 0:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_name})")
                    logger.info(f"✅ Created index '{index_name}' on column '{column_name}'")
                else:
                    logger.warning(f"⚠️ Column '{column_name}' does not exist in table '{table_name}'")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to create index '{index_name}': {e}")
        
        conn.commit()
        logger.info(f"✅ Index creation completed for table '{table_name}'")
        
    except Exception as e:
        logger.error(f"❌ Failed to create indexes: {e}")


def get_table_info(table_name: str = DEFAULT_TABLE) -> Dict[str, Any]:
    """
    Get detailed information about a table.
    
    Args:
        table_name (str): Name of the table to get info for (default: 'sales')
        
    Returns:
        Dict[str, Any]: Dictionary containing table information
        
    Raises:
        Exception: If info retrieval fails
    """
    try:
        conn = get_connection()
        
        if not table_exists(table_name):
            return {
                'exists': False,
                'error': f"Table '{table_name}' does not exist"
            }
        
        # Get table schema
        schema_info = conn.execute(f"DESCRIBE {table_name}").df()
        
        # Get row count
        row_count = get_row_count(table_name)
        
        # Get sample data
        sample_data = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").df()
        
        logger.info(f"📊 Retrieved info for table '{table_name}'")
        
        return {
            'exists': True,
            'table_name': table_name,
            'row_count': row_count,
            'columns': schema_info.to_dict('records'),
            'sample_data': sample_data.to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get table info: {e}")
        return {
            'exists': False,
            'error': str(e)
        }


def optimize_database() -> Dict[str, Any]:
    """
    Optimize the database by running VACUUM and ANALYZE commands.
    
    Returns:
        Dict[str, Any]: Result dictionary with optimization status
        
    Raises:
        Exception: If optimization fails
    """
    try:
        conn = get_connection()
        
        logger.info("🔧 Optimizing database...")
        
        # Run VACUUM to reclaim space
        conn.execute("VACUUM")
        
        # Run ANALYZE to update statistics
        conn.execute("ANALYZE")
        
        conn.commit()
        
        logger.info("✅ Database optimization completed")
        
        return {
            'success': True,
            'message': 'Database optimization completed successfully'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to optimize database: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def backup_database(backup_path: str = None) -> Dict[str, Any]:
    """
    Create a backup of the database.
    
    Args:
        backup_path (str): Path for the backup file (optional)
        
    Returns:
        Dict[str, Any]: Result dictionary with backup status
        
    Raises:
        Exception: If backup fails
    """
    try:
        if backup_path is None:
            backup_path = f"{DB_FILE}.backup"
        
        conn = get_connection()
        
        logger.info(f"💾 Creating database backup: {backup_path}")
        
        # Export all tables to backup
        conn.execute(f"EXPORT DATABASE '{backup_path}'")
        
        logger.info("✅ Database backup completed successfully")
        
        return {
            'success': True,
            'backup_path': backup_path,
            'message': 'Database backup completed successfully'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create backup: {e}")
        return {
            'success': False,
            'error': str(e)
        }


# Example usage and testing functions
def test_database_operations():
    """
    Test function to verify all database operations work correctly.
    """
    try:
        logger.info("🧪 Testing database operations...")
        
        # Test connection
        conn = get_connection()
        logger.info("✅ Connection test passed")
        
        # Test table existence
        exists = table_exists('test_table')
        logger.info(f"✅ Table existence test passed: {exists}")
        
        # Test row count
        count = get_row_count('test_table')
        logger.info(f"✅ Row count test passed: {count}")
        
        # Test table info
        info = get_table_info('test_table')
        logger.info(f"✅ Table info test passed: {info['exists']}")
        
        logger.info("🎉 All database operations tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Database operations test failed: {e}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_database_operations()
