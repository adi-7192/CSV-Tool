"""
File Management Routes - File listing, deletion, and download
Provides RESTful API for managing uploaded CSV files
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import pandas as pd
import io
import logging

from core.database import get_connection, table_exists, execute_query
from utils.error_handler import log_error

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/list")
async def list_files() -> Dict[str, Any]:
    """
    List all uploaded files with metadata
    
    Returns:
        {
            "files": [
                {
                    "filename": str,
                    "size": int,  # File size in bytes (estimated from row count)
                    "upload_date": str,  # ISO format timestamp
                    "record_count": int
                }
            ]
        }
    """
    try:
        if not table_exists('ingestion_log'):
            return {"files": []}
        
        sql = """
        SELECT
            ingestion_id,
            filename,
            uploaded_at,
            rows_inserted,
            date_range_start,
            date_range_end
        FROM ingestion_log
        ORDER BY uploaded_at DESC
        LIMIT 100
        """
        
        result_df = execute_query(sql)
        
        if result_df.empty:
            return {"files": []}
        
        files = []
        for _, row in result_df.iterrows():
            # Estimate file size (rough estimate: ~200 bytes per row)
            estimated_size = int(row.get('rows_inserted', 0) * 200)
            
            files.append({
                "filename": str(row['filename']),
                "size": estimated_size,
                "upload_date": str(row['uploaded_at']) if pd.notna(row['uploaded_at']) else None,
                "record_count": int(row['rows_inserted']) if pd.notna(row['rows_inserted']) else 0,
                "ingestion_id": str(row['ingestion_id']),  # Include for reference
                "date_range_start": str(row['date_range_start']) if pd.notna(row['date_range_start']) else None,
                "date_range_end": str(row['date_range_end']) if pd.notna(row['date_range_end']) else None,
            })
        
        logger.info(f"Listed {len(files)} files")
        return {"files": files}
    
    except Exception as e:
        log_error(e, 'list_files', {})
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@router.delete("/{filename}")
async def delete_file(filename: str) -> Dict[str, Any]:
    """
    Delete a single file by filename
    
    Args:
        filename: Name of the file to delete
    
    Returns:
        {
            "success": bool,
            "message": str,
            "deleted_rows": int
        }
    """
    try:
        if not table_exists('ingestion_log'):
            raise HTTPException(status_code=404, detail="No uploads found")
        
        conn = get_connection()
        
        # Find file by filename
        find_sql = """
        SELECT ingestion_id, filename
        FROM ingestion_log
        WHERE filename = ?
        LIMIT 1
        """
        
        result_df = conn.execute(find_sql, [filename]).fetchdf()
        
        if result_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' not found"
            )
        
        ingestion_id = result_df.iloc[0]['ingestion_id']
        found_filename = result_df.iloc[0]['filename']
        
        # Count rows to be deleted
        deleted_rows = 0
        if table_exists('sales'):
            count_sql = """
            SELECT COUNT(*) as count
            FROM sales
            WHERE ingestion_id = ?
            """
            count_df = conn.execute(count_sql, [ingestion_id]).fetchdf()
            if not count_df.empty:
                deleted_rows = int(count_df.iloc[0]['count'])
        
        # Delete rows from sales table
        if table_exists('sales') and deleted_rows > 0:
            delete_sql = """
            DELETE FROM sales
            WHERE ingestion_id = ?
            """
            conn.execute(delete_sql, [ingestion_id])
            logger.info(f"✅ Deleted {deleted_rows} rows for file: {found_filename}")
        
        # Delete from ingestion_log
        delete_log_sql = """
        DELETE FROM ingestion_log
        WHERE ingestion_id = ?
        """
        conn.execute(delete_log_sql, [ingestion_id])
        
        logger.info(f"✅ Deleted file: {found_filename} (ingestion_id: {ingestion_id})")
        
        return {
            'success': True,
            'deleted_rows': deleted_rows,
            'message': f'Successfully deleted {deleted_rows} rows for file {found_filename}'
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, 'delete_file', {'filename': filename})
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@router.delete("/all")
async def delete_all_files() -> Dict[str, Any]:
    """
    Delete all uploaded files and data
    
    WARNING: This will delete ALL data in the database!
    
    Returns:
        {
            "success": bool,
            "message": str
        }
    """
    try:
        from services.uploads_service import delete_all_uploads
        
        result = delete_all_uploads()
        
        if not result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=result.get('message', 'Failed to delete all files')
            )
        
        logger.warning("⚠️  All files deleted via /api/files/all endpoint")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, 'delete_all_files', {})
        raise HTTPException(status_code=500, detail=f"Error deleting all files: {str(e)}")


@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a file by filename as CSV
    
    Args:
        filename: Name of the file to download
    
    Returns:
        CSV file download
    """
    try:
        if not table_exists('ingestion_log'):
            raise HTTPException(status_code=404, detail="No uploads found")
        
        conn = get_connection()
        
        # Find file by filename
        find_sql = """
        SELECT ingestion_id, filename
        FROM ingestion_log
        WHERE filename = ?
        LIMIT 1
        """
        
        result_df = conn.execute(find_sql, [filename]).fetchdf()
        
        if result_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' not found"
            )
        
        ingestion_id = result_df.iloc[0]['ingestion_id']
        found_filename = result_df.iloc[0]['filename']
        
        # Export data for this ingestion_id from sales table
        if not table_exists('sales'):
            raise HTTPException(status_code=404, detail="No data found for this file")
        
        export_sql = """
        SELECT *
        FROM sales
        WHERE ingestion_id = ?
        ORDER BY loaded_at
        """
        
        data_df = conn.execute(export_sql, [ingestion_id]).fetchdf()
        
        if data_df.empty:
            raise HTTPException(status_code=404, detail="No data found for this file")
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        data_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Create streaming response
        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{found_filename}"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, 'download_file', {'filename': filename})
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

