"""
Database Reset Utility - Reset database to clean state

Use this to reset the database if you need to start fresh with standardized column names.
"""
import logging
from core.database import get_connection, table_exists, execute_query

logger = logging.getLogger(__name__)


def reset_database(table_name: str = 'sales') -> dict:
    """
    Reset database by dropping and recreating table
    
    WARNING: This will delete all data!
    
    Returns:
        dict: Result with success status
    """
    try:
        conn = get_connection()
        
        if table_exists(table_name):
            logger.warning(f"⚠️  Dropping existing table '{table_name}' - ALL DATA WILL BE LOST!")
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            logger.info(f"✅ Dropped table '{table_name}'")
        
        # Also reset ingestion log if it exists
        if table_exists('ingestion_log'):
            logger.info(f"Dropping ingestion_log table...")
            conn.execute("DROP TABLE IF EXISTS ingestion_log")
        
        conn.commit()
        
        logger.info(f"✅ Database reset complete. Table '{table_name}' has been dropped.")
        logger.info("📋 Next upload will create table with standardized column names.")
        
        return {
            'success': True,
            'message': f'Database reset complete. Table {table_name} dropped.',
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to reset database: {e}")
        return {
            'success': False,
            'error': str(e),
        }


def check_database_schema(table_name: str = 'sales') -> dict:
    """
    Check current database schema and report column naming issues
    
    Returns:
        dict: Schema analysis with recommendations
    """
    if not table_exists(table_name):
        return {
            'exists': False,
            'message': f'Table {table_name} does not exist',
        }
    
    try:
        table_info = execute_query(f"DESCRIBE {table_name}")
        columns = table_info['column_name'].tolist()
        
        # Check for old vs standardized column names
        old_column_patterns = ['Invoice Amount', 'Invoice Date', 'Invoice Number', 'Transaction Type']
        standardized_patterns = ['revenue_amount', 'order_date', 'order_id', 'transaction_type']
        
        old_columns_found = [col for col in columns if any(pattern in col for pattern in old_column_patterns)]
        standardized_columns_found = [col for col in columns if col in standardized_patterns]
        
        has_lineage = all(col in columns for col in ['source_file', 'ingestion_id', 'loaded_at'])
        
        analysis = {
            'exists': True,
            'total_columns': len(columns),
            'columns': columns,
            'has_old_column_names': len(old_columns_found) > 0,
            'old_columns': old_columns_found,
            'has_standardized_names': len(standardized_columns_found) > 0,
            'standardized_columns': standardized_columns_found,
            'has_lineage_columns': has_lineage,
            'lineage_columns': ['source_file', 'ingestion_id', 'loaded_at'] if has_lineage else [],
            'recommendation': None,
        }
        
        # Generate recommendation
        if old_columns_found and not standardized_columns_found:
            analysis['recommendation'] = 'RESET_DATABASE'
            analysis['message'] = 'Database has old column names. Reset database and re-upload CSVs to get standardized names.'
        elif old_columns_found and standardized_columns_found:
            analysis['recommendation'] = 'MIGRATE_COLUMNS'
            analysis['message'] = 'Database has both old and new column names. Consider migrating or resetting.'
        elif standardized_columns_found and not old_columns_found:
            analysis['recommendation'] = 'OK'
            analysis['message'] = 'Database has standardized column names. No action needed.'
        
        if not has_lineage:
            analysis['recommendation'] = 'RESET_DATABASE'
            analysis['message'] = 'Database missing lineage columns. Reset and re-upload.'
        
        return analysis
    
    except Exception as e:
        logger.error(f"Failed to check schema: {e}")
        return {
            'exists': True,
            'error': str(e),
        }

