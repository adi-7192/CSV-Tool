#!/usr/bin/env python3
"""
Clear Database Script

Clears all data from the DuckDB database for testing purposes.
This script will:
1. Drop the 'sales' table (main transaction data)
2. Drop the 'ingestion_log' table (upload history)
3. Recreate empty tables with proper schema
4. Close and reset the database connection

Usage:
    python backend/scripts/clear_database.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from services.database_reset import reset_database
from core.database import close_connection, init_database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_database():
    """Clear all data from the database"""
    try:
        logger.info("🗑️  Clearing database...")
        
        # Use the existing reset_database function
        result = reset_database('sales')
        
        if not result.get('success'):
            logger.error(f"❌ Failed to reset database: {result.get('error')}")
            return False
        
        logger.info("✅ Dropped 'sales' table")
        
        # Also drop ingestion_log table
        try:
            from core.database import get_connection, table_exists
            conn = get_connection()
            if table_exists('ingestion_log'):
                conn.execute("DROP TABLE IF EXISTS ingestion_log")
                logger.info("✅ Dropped 'ingestion_log' table")
        except Exception as e:
            logger.warning(f"⚠️  Could not drop 'ingestion_log' table: {e}")
        
        # Close connection to release lock
        close_connection()
        
        # Reinitialize database (creates empty tables)
        logger.info("🔄 Reinitializing database with empty tables...")
        init_database()
        
        logger.info("✅ Database cleared successfully!")
        logger.info("📊 Database is now empty and ready for new data uploads")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to clear database: {e}")
        logger.error("💡 Tip: Make sure the backend server is stopped before running this script")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🗑️  DATABASE CLEAR SCRIPT")
    print("="*60 + "\n")
    
    success = clear_database()
    
    if success:
        print("\n" + "="*60)
        print("✅ Database cleared successfully!")
        print("📊 You can now upload new CSV files to test the changes")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ Failed to clear database")
        print("="*60 + "\n")
        sys.exit(1)

