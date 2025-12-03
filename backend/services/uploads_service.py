"""
Uploads Service - Manage upload metadata and file operations
Tracks uploads in metadata table and provides file management functions
"""
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from core.database import get_connection, table_exists, get_row_count
from utils.error_handler import log_error

logger = logging.getLogger(__name__)

# =============================================================================
# METADATA TABLE MANAGEMENT
# =============================================================================

def init_uploads_metadata_table():
    """
    Create uploads_metadata table if it doesn't exist
    Tracks each file upload with metadata
    """
    conn = get_connection()
    
    if not table_exists('uploads_metadata'):
        create_table_sql = """
        CREATE TABLE uploads_metadata (
            file_id VARCHAR PRIMARY KEY,
            ingestion_id VARCHAR,
            filename VARCHAR NOT NULL,
            row_count INTEGER NOT NULL,
            file_size BIGINT NOT NULL,
            date_range_start DATE,
            date_range_end DATE,
            column_names VARCHAR,
            upload_timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        conn.execute(create_table_sql)
        logger.info("✅ Created uploads_metadata table")
    else:
        logger.debug("uploads_metadata table already exists")


def create_upload_metadata(
    ingestion_id: str,
    filename: str,
    row_count: int,
    file_size: int,
    date_range_start: Optional[str] = None,
    date_range_end: Optional[str] = None,
    column_names: Optional[List[str]] = None,
) -> str:
    """
    Create metadata entry for an upload
    
    Returns: file_id (UUID)
    """
    init_uploads_metadata_table()
    
    file_id = str(uuid.uuid4())
    conn = get_connection()
    
    column_names_str = ','.join(column_names) if column_names else None
    
    insert_sql = """
    INSERT INTO uploads_metadata (
        file_id, ingestion_id, filename, row_count, file_size,
        date_range_start, date_range_end, column_names, upload_timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    conn.execute(insert_sql, (
        file_id,
        ingestion_id,
        filename,
        row_count,
        file_size,
        date_range_start,
        date_range_end,
        column_names_str,
        datetime.now(),
    ))
    
    logger.info(f"✅ Created upload metadata: file_id={file_id}, filename={filename}, rows={row_count}")
    return file_id


def backfill_metadata_from_existing_data() -> int:
    """
    Backfill metadata table from existing sales data
    Groups data by ingestion_id and creates metadata entries
    
    Returns: Number of metadata entries created
    """
    init_uploads_metadata_table()
    
    if not table_exists('sales'):
        return 0
    
    try:
        conn = get_connection()
        
        # Check if metadata already exists
        metadata_check = conn.execute("SELECT COUNT(*) as count FROM uploads_metadata").fetchdf()
        if not metadata_check.empty and metadata_check.iloc[0]['count'] > 0:
            logger.debug("Metadata already exists, skipping backfill")
            return 0
        
        # Check if sales table has ingestion_id column
        schema_df = conn.execute("DESCRIBE sales").fetchdf()
        has_ingestion_id = 'ingestion_id' in schema_df['column_name'].values
        
        # Get date column
        date_columns = ['order_date', 'Order Date', 'Invoice Date', 'invoice_date']
        date_col = None
        for col in date_columns:
            if col in schema_df['column_name'].values:
                date_col = col
                break
        
        if has_ingestion_id:
            # Group by ingestion_id
            sql = """
            SELECT 
                ingestion_id,
                COUNT(*) as row_count,
                MIN(CAST(? AS DATE)) as min_date,
                MAX(CAST(? AS DATE)) as max_date
            FROM sales
            WHERE ingestion_id IS NOT NULL AND ingestion_id != ''
            GROUP BY ingestion_id
            """
            if date_col:
                sql = sql.replace('?', f'"{date_col}"')
            else:
                sql = sql.replace('CAST(? AS DATE)', '1')  # Dummy replacement
            
            df = conn.execute(sql).fetchdf()
            
            created_count = 0
            for _, row in df.iterrows():
                ingestion_id = row['ingestion_id']
                row_count = int(row['row_count'])
                
                # Create metadata entry
                file_id = str(uuid.uuid4())
                filename = f"upload_{ingestion_id[:8]}.csv"  # Use first 8 chars of ingestion_id
                file_size = row_count * 100  # Estimate: ~100 bytes per row
                
                date_range_start = None
                date_range_end = None
                if date_col and row.get('min_date') and row.get('max_date'):
                    date_range_start = str(row['min_date']).split(' ')[0]
                    date_range_end = str(row['max_date']).split(' ')[0]
                
                insert_sql = """
                INSERT INTO uploads_metadata (
                    file_id, ingestion_id, filename, row_count, file_size,
                    date_range_start, date_range_end, column_names, upload_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                conn.execute(insert_sql, (
                    file_id,
                    ingestion_id,
                    filename,
                    row_count,
                    file_size,
                    date_range_start,
                    date_range_end,
                    None,  # Column names not available
                    datetime.now(),
                ))
                created_count += 1
            
            if created_count > 0:
                logger.info(f"✅ Backfilled {created_count} metadata entries from existing data")
            return created_count
        else:
            # No ingestion_id - create a single synthetic entry for all data
            total_rows = get_row_count('sales')
            if total_rows == 0:
                return 0
            
            # Get date range
            date_range_start = None
            date_range_end = None
            if date_col:
                try:
                    date_sql = f"""
                    SELECT 
                        MIN(CAST("{date_col}" AS DATE)) as min_date,
                        MAX(CAST("{date_col}" AS DATE)) as max_date
                    FROM sales
                    WHERE "{date_col}" IS NOT NULL AND "{date_col}" != ''
                    """
                    date_df = conn.execute(date_sql).fetchdf()
                    if not date_df.empty:
                        if date_df.iloc[0]['min_date']:
                            date_range_start = str(date_df.iloc[0]['min_date']).split(' ')[0]
                        if date_df.iloc[0]['max_date']:
                            date_range_end = str(date_df.iloc[0]['max_date']).split(' ')[0]
                except Exception as e:
                    logger.warning(f"Could not get date range: {e}")
            
            # Create single metadata entry
            file_id = str(uuid.uuid4())
            filename = "existing_data.csv"
            file_size = total_rows * 100  # Estimate
            
            insert_sql = """
            INSERT INTO uploads_metadata (
                file_id, ingestion_id, filename, row_count, file_size,
                date_range_start, date_range_end, column_names, upload_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            conn.execute(insert_sql, (
                file_id,
                None,  # No ingestion_id
                filename,
                total_rows,
                file_size,
                date_range_start,
                date_range_end,
                None,
                datetime.now(),
            ))
            
            logger.info(f"✅ Created synthetic metadata entry for {total_rows} existing rows")
            return 1
            
    except Exception as e:
        log_error(e, 'backfill_metadata_from_existing_data', {})
        logger.error(f"Error backfilling metadata: {e}")
        return 0


def get_all_uploads() -> List[Dict[str, Any]]:
    """
    Get list of all uploads with metadata
    Automatically backfills metadata if missing but data exists
    
    Returns: List of upload dictionaries
    """
    init_uploads_metadata_table()
    
    if not table_exists('uploads_metadata'):
        # Try to backfill if sales table exists
        if table_exists('sales'):
            backfill_metadata_from_existing_data()
        else:
            return []
    
    try:
        sql = """
        SELECT 
            file_id,
            ingestion_id,
            filename,
            row_count,
            file_size,
            date_range_start,
            date_range_end,
            upload_timestamp,
            created_at
        FROM uploads_metadata
        ORDER BY upload_timestamp DESC
        """
        
        conn = get_connection()
        df = conn.execute(sql).fetchdf()
        
        # If metadata is empty but sales table has data, backfill
        if df.empty and table_exists('sales') and get_row_count('sales') > 0:
            backfill_metadata_from_existing_data()
            # Fetch again after backfill
            df = conn.execute(sql).fetchdf()
        
        if df.empty:
            return []
        
        uploads = []
        for _, row in df.iterrows():
            uploads.append({
                'file_id': row['file_id'],
                'ingestion_id': row.get('ingestion_id'),
                'filename': row['filename'],
                'row_count': int(row['row_count']),
                'file_size': int(row['file_size']),
                'date_range': {
                    'start': str(row['date_range_start']) if row.get('date_range_start') else None,
                    'end': str(row['date_range_end']) if row.get('date_range_end') else None,
                },
                'upload_timestamp': row['upload_timestamp'].isoformat() if hasattr(row['upload_timestamp'], 'isoformat') else str(row['upload_timestamp']),
            })
        
        return uploads
    except Exception as e:
        log_error(e, 'get_all_uploads', {})
        logger.error(f"Error fetching uploads: {e}")
        return []


def get_upload_details(file_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed metadata for a specific upload
    
    Returns: Upload details dictionary or None if not found
    """
    init_uploads_metadata_table()
    
    if not table_exists('uploads_metadata'):
        return None
    
    try:
        sql = """
        SELECT 
            file_id,
            ingestion_id,
            filename,
            row_count,
            file_size,
            date_range_start,
            date_range_end,
            column_names,
            upload_timestamp,
            created_at
        FROM uploads_metadata
        WHERE file_id = ?
        """
        
        conn = get_connection()
        df = conn.execute(sql, (file_id,)).fetchdf()
        
        if df.empty:
            return None
        
        row = df.iloc[0]
        
        # Get sample data from sales table
        sample_data = []
        if table_exists('sales'):
            try:
                ingestion_id = row.get('ingestion_id')
                if ingestion_id:
                    sample_sql = """
                    SELECT *
                    FROM sales
                    WHERE ingestion_id = ?
                    LIMIT 5
                    """
                    sample_df = conn.execute(sample_sql, (ingestion_id,)).fetchdf()
                    if not sample_df.empty:
                        # Convert to dict list
                        sample_data = sample_df.to_dict('records')
            except Exception as e:
                logger.warning(f"Could not fetch sample data: {e}")
        
        column_names = []
        if row.get('column_names'):
            column_names = row['column_names'].split(',')
        
        return {
            'file_id': row['file_id'],
            'ingestion_id': row.get('ingestion_id'),
            'filename': row['filename'],
            'row_count': int(row['row_count']),
            'file_size': int(row['file_size']),
            'date_range': {
                'start': str(row['date_range_start']) if row.get('date_range_start') else None,
                'end': str(row['date_range_end']) if row.get('date_range_end') else None,
            },
            'column_names': column_names,
            'upload_timestamp': row['upload_timestamp'].isoformat() if hasattr(row['upload_timestamp'], 'isoformat') else str(row['upload_timestamp']),
            'sample_data': sample_data,
        }
    except Exception as e:
        log_error(e, 'get_upload_details', {'file_id': file_id})
        logger.error(f"Error fetching upload details: {e}")
        return None


def delete_upload_data(file_id: str) -> Dict[str, Any]:
    """
    Delete data associated with a specific file
    
    Returns: {success: bool, deleted_rows: int, message: str}
    """
    init_uploads_metadata_table()
    
    if not table_exists('uploads_metadata'):
        return {
            'success': False,
            'deleted_rows': 0,
            'message': 'No uploads metadata found'
        }
    
    try:
        # Get ingestion_id from metadata
        sql = """
        SELECT ingestion_id, filename
        FROM uploads_metadata
        WHERE file_id = ?
        """
        
        conn = get_connection()
        df = conn.execute(sql, (file_id,)).fetchdf()
        
        if df.empty:
            return {
                'success': False,
                'deleted_rows': 0,
                'message': f'File with file_id {file_id} not found'
            }
        
        ingestion_id = df.iloc[0]['ingestion_id']
        filename = df.iloc[0]['filename']
        
        # Count rows to be deleted
        deleted_rows = 0
        if table_exists('sales') and ingestion_id:
            count_sql = """
            SELECT COUNT(*) as count
            FROM sales
            WHERE ingestion_id = ?
            """
            count_df = conn.execute(count_sql, (ingestion_id,)).fetchdf()
            if not count_df.empty:
                deleted_rows = int(count_df.iloc[0]['count'])
        
        # Delete rows from sales table
        if table_exists('sales') and ingestion_id:
            delete_sql = """
            DELETE FROM sales
            WHERE ingestion_id = ?
            """
            conn = get_connection()
            conn.execute(delete_sql, (ingestion_id,))
            logger.info(f"✅ Deleted {deleted_rows} rows for file_id={file_id}, filename={filename}")
        
        # Delete metadata entry
        delete_metadata_sql = """
        DELETE FROM uploads_metadata
        WHERE file_id = ?
        """
        conn.execute(delete_metadata_sql, (file_id,))
        
        return {
            'success': True,
            'deleted_rows': deleted_rows,
            'message': f'Successfully deleted {deleted_rows} rows for file {filename}'
        }
    except Exception as e:
        log_error(e, 'delete_upload_data', {'file_id': file_id})
        logger.error(f"Error deleting upload data: {e}")
        return {
            'success': False,
            'deleted_rows': 0,
            'message': f'Error deleting upload data: {str(e)}'
        }


def delete_all_uploads() -> Dict[str, Any]:
    """
    Delete all uploaded data and metadata
    
    Returns: {success: bool, message: str}
    """
    init_uploads_metadata_table()
    
    try:
        conn = get_connection()
        
        # Drop sales table
        if table_exists('sales'):
            conn.execute("DROP TABLE IF EXISTS sales")
            logger.info("✅ Dropped sales table")
        
        # Drop uploads_metadata table
        if table_exists('uploads_metadata'):
            conn.execute("DROP TABLE IF EXISTS uploads_metadata")
            logger.info("✅ Dropped uploads_metadata table")
        
        # Drop ingestion_log table if it exists
        if table_exists('ingestion_log'):
            conn.execute("DROP TABLE IF EXISTS ingestion_log")
            logger.info("✅ Dropped ingestion_log table")
        
        # Re-initialize database to ensure tables can be recreated on next upload
        from core.database import init_database
        try:
            init_database()
            logger.info("✅ Database re-initialized after deletion")
        except Exception as init_error:
            logger.warning(f"Could not re-initialize database: {init_error}")
            # This is not critical - tables will be created on next upload
        
        return {
            'success': True,
            'message': 'All data deleted successfully'
        }
    except Exception as e:
        log_error(e, 'delete_all_uploads', {})
        logger.error(f"Error deleting all uploads: {e}")
        return {
            'success': False,
            'message': f'Error deleting all data: {str(e)}'
        }

