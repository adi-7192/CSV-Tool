"""
Upload API Endpoints - Handle CSV file uploads
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Dict, Any, List
import pandas as pd

from services.upload_service import process_csv_upload
from models.responses import UploadResponse
from services.database_reset import reset_database, check_database_schema
from api.deps.auth_deps import get_current_user
from models.user import UserInDB

router = APIRouter()


@router.post("/csv", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(..., description="CSV file to upload")
):
    """
    Upload and process CSV file
    
    Returns:
        - ingestion_id: Unique ID for this upload
        - filename: Original filename
        - rows_uploaded: Number of rows processed
        - duplicates_removed: Number of duplicate rows removed
        - validation_report: Data quality report
        - column_mapping: Detected column mapping
    """
    
    # Validate file type
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    # Validate file size (100MB max)
    file_content = await file.read()
    
    if len(file_content) > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 100MB"
        )
    
    # Process upload
    result = process_csv_upload(file_content, file.filename)
    
    if not result.get('success'):
        raise HTTPException(
            status_code=500,
            detail=result.get('error', 'Upload processing failed')
        )
    
    # Shape response to include concise quality warnings
    return {
        'success': True,
        'rows_processed': result.get('rows_processed', result.get('rows_raw', 0)),
        'rows_inserted': result.get('rows_inserted', result.get('rows_uploaded', 0)),
        'validation_warnings': result.get('validation_warnings', []),
        'ingestion_id': result.get('ingestion_id'),
        'filename': result.get('filename'),
    }


@router.get("/history")
async def get_upload_history(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get upload history from ingestion_log table (tenant-isolated).
    
    TENANT ISOLATION: Only returns uploads belonging to the current user's tenant.
    
    Returns list of uploads with metadata:
    - ingestion_id
    - filename
    - uploaded_at
    - rows_inserted
    - date_range_start
    - date_range_end
    - validation_status
    - file_size
    - file_hash
    """
    from core.database import get_connection, table_exists
    from utils.tenant_filter import get_tenant_filter_sql
    
    # TENANT ISOLATION: Get tenant_id from authenticated user
    tenant_id = current_user.tenant_id or str(current_user.id)
    
    if not table_exists('ingestion_log'):
        return {"uploads": []}
    
    try:
        conn = get_connection()
        tenant_filter = get_tenant_filter_sql(tenant_id)
        
        # Get uploads from ingestion_log, but only show entries that have associated sales data
        # This filters out orphaned ingestion_log entries where sales data was already deleted
        if table_exists('sales'):
            # Build tenant filter conditions for both tables
            safe_tenant_id = tenant_id.replace("'", "''")
            ingestion_tenant_condition = f'il.tenant_id = \'{safe_tenant_id}\''
            sales_tenant_condition = f's.tenant_id = \'{safe_tenant_id}\''
            
            sql = f"""
            SELECT DISTINCT
                il.ingestion_id,
                il.filename,
                il.uploaded_at,
                il.rows_inserted,
                il.date_range_start,
                il.date_range_end,
                il.validation_status,
                il.file_size,
                il.file_hash,
                il.processing_time_seconds
            FROM ingestion_log il
            WHERE {ingestion_tenant_condition}
            AND EXISTS (
                SELECT 1
                FROM sales s
                WHERE s.ingestion_id = il.ingestion_id
                AND {sales_tenant_condition}
                LIMIT 1
            )
            ORDER BY il.uploaded_at DESC
            LIMIT 100
            """
        else:
            # If sales table doesn't exist, fall back to showing all ingestion_log entries
            sql = f"""
            SELECT
                ingestion_id,
                filename,
                uploaded_at,
                rows_inserted,
                date_range_start,
                date_range_end,
                validation_status,
                file_size,
                file_hash,
                processing_time_seconds
            FROM ingestion_log
            WHERE {tenant_filter}
            ORDER BY uploaded_at DESC
            LIMIT 100
            """
        
        result_df = conn.execute(sql).fetchdf()
        
        if result_df.empty:
            return {"uploads": []}
        
        # Convert to list of dicts
        uploads = result_df.to_dict('records')
        
        # Format dates as strings
        for upload in uploads:
            if 'uploaded_at' in upload and pd.notna(upload['uploaded_at']):
                upload['uploaded_at'] = str(upload['uploaded_at'])
            if 'date_range_start' in upload and pd.notna(upload['date_range_start']):
                upload['date_range_start'] = str(upload['date_range_start'])
            if 'date_range_end' in upload and pd.notna(upload['date_range_end']):
                upload['date_range_end'] = str(upload['date_range_end'])
        
        return {"uploads": uploads}
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error fetching upload history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/all")
async def delete_all_files(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Delete all uploaded files and data for the current tenant (tenant-isolated).
    
    TENANT ISOLATION: Only deletes files belonging to the current user's tenant.
    
    Returns:
        {
            "success": bool,
            "message": str,
            "deleted_rows": int
        }
    """
    from core.database import get_connection, table_exists
    from utils.tenant_filter import get_tenant_filter_sql
    from services.embedding_service import get_chroma_client
    
    # TENANT ISOLATION: Get tenant_id from authenticated user
    tenant_id = current_user.tenant_id or str(current_user.id)
    
    try:
        conn = get_connection()
        tenant_filter = get_tenant_filter_sql(tenant_id)
        
        # Count rows to be deleted
        deleted_rows = 0
        if table_exists('sales'):
            count_sql = f"SELECT COUNT(*) as count FROM sales WHERE {tenant_filter}"
            count_df = conn.execute(count_sql).fetchdf()
            if not count_df.empty:
                deleted_rows = int(count_df.iloc[0]['count'])
        
        # 1. Delete all sales rows for this tenant
        if table_exists('sales') and deleted_rows > 0:
            delete_sql = f"DELETE FROM sales WHERE {tenant_filter}"
            conn.execute(delete_sql)
        
        # 2. Delete ingestion logs for this tenant
        if table_exists('ingestion_log'):
            delete_log_sql = f"DELETE FROM ingestion_log WHERE {tenant_filter}"
            conn.execute(delete_log_sql)
        
        # 3. Delete tenant's ChromaDB collection
        try:
            chroma_client = get_chroma_client()
            collection_name = f"rag_{tenant_id}"
            try:
                collection = chroma_client.get_collection(collection_name)
                collection.delete()  # Delete all documents in collection
            except Exception:
                # Collection doesn't exist, which is fine
                pass
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not delete ChromaDB collection: {e}")
        
        return {
            'success': True,
            'deleted_rows': deleted_rows,
            'message': f'Successfully deleted all data ({deleted_rows} rows)'
        }
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error deleting all files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting all files: {str(e)}")


@router.delete("/{ingestion_id}")
async def delete_file(
    ingestion_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Delete a specific uploaded file and its data (tenant-isolated).
    
    TENANT ISOLATION: Only deletes files belonging to the current user's tenant.
    Users cannot delete other tenants' files even if they know the ingestion_id.
    
    Args:
        ingestion_id: The ingestion ID of the file to delete
        current_user: Authenticated user (from JWT token)
    
    Returns:
        {
            "success": bool,
            "deleted_rows": int,
            "message": str
        }
    """
    from core.database import get_connection, table_exists, execute_query
    from utils.tenant_filter import get_tenant_filter_sql
    
    # TENANT ISOLATION: Get tenant_id from authenticated user
    tenant_id = current_user.tenant_id or str(current_user.id)
    
    if not table_exists('ingestion_log'):
        raise HTTPException(status_code=404, detail="No uploads found")
    
    try:
        conn = get_connection()
        tenant_filter = get_tenant_filter_sql(tenant_id)
        
        # Get file info from ingestion_log - verify it belongs to this tenant
        # Check ingestion_log FIRST (not sales), because ingestion_log entry might exist even if sales data was already deleted
        sql = f"""
        SELECT ingestion_id, filename
        FROM ingestion_log
        WHERE ingestion_id = ? AND {tenant_filter}
        LIMIT 1
        """
        
        result_df = conn.execute(sql, [ingestion_id]).fetchdf()
        
        if result_df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"File with ingestion_id {ingestion_id} not found or does not belong to your account"
            )
        
        filename = result_df.iloc[0]['filename']
        
        # Count rows to be deleted (TENANT ISOLATED)
        # Note: Sales rows might already be deleted, so we check but don't require them to exist
        deleted_rows = 0
        if table_exists('sales'):
            count_sql = f"""
            SELECT COUNT(*) as count
            FROM sales
            WHERE ingestion_id = ? AND {tenant_filter}
            """
            count_df = conn.execute(count_sql, [ingestion_id]).fetchdf()
            if not count_df.empty:
                deleted_rows = int(count_df.iloc[0]['count'])
        
        # Delete rows from sales table (TENANT ISOLATED)
        # Delete even if count is 0 (in case of race conditions or partial deletions)
        if table_exists('sales'):
            delete_sql = f"""
            DELETE FROM sales
            WHERE ingestion_id = ? AND {tenant_filter}
            """
            conn.execute(delete_sql, [ingestion_id])
        
        # Always delete from ingestion_log (tenant-isolated)
        # This ensures the entry is removed even if sales data was already deleted
        delete_log_sql = f"""
        DELETE FROM ingestion_log
        WHERE ingestion_id = ? AND {tenant_filter}
        """
        conn.execute(delete_log_sql, [ingestion_id])
        
        return {
            'success': True,
            'deleted_rows': deleted_rows,
            'message': f'Successfully deleted {deleted_rows} rows for file {filename}'
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@router.get("/download/{ingestion_id}")
async def download_file(ingestion_id: str):
    """
    Download the original CSV file for a specific upload
    
    Args:
        ingestion_id: The ingestion ID of the file to download
    
    Returns:
        CSV file download
    """
    from fastapi.responses import StreamingResponse
    from core.database import execute_query, table_exists
    import io
    
    # Get file info from ingestion_log
    if not table_exists('ingestion_log'):
        raise HTTPException(status_code=404, detail="No uploads found")
    
    try:
        conn = get_connection()
        
        # Get file info from ingestion_log
        sql = """
        SELECT filename, ingestion_id
        FROM ingestion_log
        WHERE ingestion_id = ?
        LIMIT 1
        """
        
        result_df = conn.execute(sql, [ingestion_id]).fetchdf()
        
        if result_df.empty:
            raise HTTPException(status_code=404, detail=f"File with ingestion_id {ingestion_id} not found")
        
        filename = result_df.iloc[0]['filename']
        
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
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@router.post("/multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    """
    Upload multiple CSV files
    
    Returns:
        - results: List of upload results for each file
        - total_files: Total number of files processed
        - successful: Number of successful uploads
        - failed: Number of failed uploads
    """
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            # Validate file type
            if not file.filename or not file.filename.endswith('.csv'):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'Only CSV files are supported'
                })
                failed += 1
                continue
            
            # Read file content
            file_content = await file.read()
            
            # Validate file size
            if len(file_content) > 100 * 1024 * 1024:  # 100MB
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'File too large. Maximum size is 100MB'
                })
                failed += 1
                continue
            
            # Process upload
            result = process_csv_upload(file_content, file.filename)
            
            if result.get('success'):
                successful += 1
            else:
                failed += 1
            
            results.append(result)
        
        except Exception as e:
            results.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })
            failed += 1
    
    return {
        "results": results,
        "total_files": len(files),
        "successful": successful,
        "failed": failed
    }


@router.post("/reset-database")
async def reset_upload_database():
    """
    Reset database by dropping sales table
    
    WARNING: This will delete ALL data!
    
    Use this before re-uploading CSVs to ensure standardized column names.
    
    Recommended workflow:
    1. Reset database (this endpoint)
    2. Re-upload all CSV files
    3. Verify schema with /api/upload/check-schema
    """
    result = reset_database('sales')
    
    if not result.get('success'):
        raise HTTPException(
            status_code=500,
            detail=result.get('error', 'Failed to reset database')
        )
    
    return {
        'success': True,
        'message': result.get('message', 'Database reset complete'),
        'warning': 'All data has been deleted. Re-upload CSV files to populate with standardized column names.'
    }


@router.get("/verify-data-integrity")
async def verify_data_integrity():
    """
    Verify data integrity after CSV re-upload
    
    Checks:
    1. Count records per source file
    2. Check for duplicates (by order_id and business key)
    3. Verify Order ID mapping (empty/synthetic IDs)
    4. Check revenue calculations
    5. Compare with expected values
    
    Returns:
        dict: Verification results with issues and recommendations
    """
    from core.database import get_connection, execute_query, table_exists
    
    results = {
        'table_exists': False,
        'total_records': 0,
        'records_by_file': {},
        'duplicate_check': {},
        'order_id_check': {},
        'revenue_check': {},
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    if not table_exists('sales'):
        results['issues'].append("❌ Table 'sales' does not exist. Database may be empty.")
        return results
    
    results['table_exists'] = True
    
    try:
        # 1. Count total records
        total_query = "SELECT COUNT(*) as total FROM sales"
        total_df = execute_query(total_query)
        results['total_records'] = int(total_df['total'].iloc[0]) if not total_df.empty else 0
        
        # 2. Count records per source file
        file_count_query = """
        SELECT 
            source_file,
            COUNT(*) as record_count,
            COUNT(DISTINCT order_id) as unique_order_ids,
            MIN(loaded_at) as first_upload,
            MAX(loaded_at) as last_upload
        FROM sales
        GROUP BY source_file
        ORDER BY source_file
        """
        file_counts_df = execute_query(file_count_query)
        
        if not file_counts_df.empty:
            for _, row in file_counts_df.iterrows():
                file_name = row['source_file']
                count = int(row['record_count'])
                unique_ids = int(row['unique_order_ids'])
                
                results['records_by_file'][file_name] = {
                    'total_records': count,
                    'unique_order_ids': unique_ids,
                    'duplicates': count - unique_ids,
                    'first_upload': str(row['first_upload']),
                    'last_upload': str(row['last_upload'])
                }
                
                if count != unique_ids:
                    results['warnings'].append(
                        f"⚠️  {file_name}: {count - unique_ids:,} duplicate order IDs found"
                    )
        else:
            results['warnings'].append("⚠️  No source_file information found")
        
        # 3. Check for duplicates (by order_id + transaction_type)
        duplicate_check_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT order_id) as unique_order_ids,
            COUNT(DISTINCT CONCAT(order_id, '|', transaction_type)) as unique_business_keys
        FROM sales
        """
        dup_check_df = execute_query(duplicate_check_query)
        
        if not dup_check_df.empty:
            total = int(dup_check_df['total_records'].iloc[0])
            unique_order_ids = int(dup_check_df['unique_order_ids'].iloc[0])
            unique_business_keys = int(dup_check_df['unique_business_keys'].iloc[0])
            
            results['duplicate_check'] = {
                'total_records': total,
                'unique_order_ids': unique_order_ids,
                'unique_business_keys': unique_business_keys,
                'duplicate_order_ids': total - unique_order_ids,
                'duplicate_business_keys': total - unique_business_keys
            }
            
            if total > unique_order_ids:
                dup_count = total - unique_order_ids
                results['issues'].append(
                    f"❌ DUPLICATES FOUND: {dup_count:,} records have duplicate order_ids"
                )
                
                # Show sample duplicates
                sample_dup_query = """
                SELECT 
                    order_id,
                    transaction_type,
                    COUNT(*) as duplicate_count,
                    STRING_AGG(DISTINCT source_file, ', ') as source_files
                FROM sales
                GROUP BY order_id, transaction_type
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC
                LIMIT 10
                """
                sample_dup_df = execute_query(sample_dup_query)
                if not sample_dup_df.empty:
                    results['duplicate_check']['sample_duplicates'] = sample_dup_df.to_dict('records')
            
            if total > unique_business_keys:
                dup_bk_count = total - unique_business_keys
                results['issues'].append(
                    f"❌ DUPLICATE BUSINESS KEYS: {dup_bk_count:,} records have duplicate (order_id + transaction_type)"
                )
        
        # 4. Verify Order ID mapping
        order_id_check_query = """
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN order_id IS NULL OR order_id = '' THEN 1 END) as empty_order_ids,
            COUNT(CASE WHEN order_id LIKE 'UNKNOWN_%' THEN 1 END) as synthetic_order_ids,
            COUNT(CASE WHEN order_id NOT LIKE 'UNKNOWN_%' AND order_id IS NOT NULL AND order_id != '' THEN 1 END) as real_order_ids
        FROM sales
        """
        order_id_df = execute_query(order_id_check_query)
        
        if not order_id_df.empty:
            total = int(order_id_df['total'].iloc[0])
            empty = int(order_id_df['empty_order_ids'].iloc[0])
            synthetic = int(order_id_df['synthetic_order_ids'].iloc[0])
            real = int(order_id_df['real_order_ids'].iloc[0])
            
            results['order_id_check'] = {
                'total_records': total,
                'empty_order_ids': empty,
                'synthetic_order_ids': synthetic,
                'real_order_ids': real
            }
            
            if empty > 0:
                results['issues'].append(
                    f"❌ {empty:,} records have empty order_ids"
                )
            
            if synthetic > 0:
                results['warnings'].append(
                    f"⚠️  {synthetic:,} records have synthetic order IDs (UNKNOWN_*)"
                )
        
        # 5. Check revenue calculations
        revenue_query = """
        SELECT 
            transaction_type,
            COUNT(*) as transaction_count,
            SUM(ABS(revenue_amount)) as total_revenue,
            AVG(ABS(revenue_amount)) as avg_revenue
        FROM sales
        WHERE revenue_amount IS NOT NULL
        GROUP BY transaction_type
        ORDER BY transaction_type
        """
        revenue_df = execute_query(revenue_query)
        
        if not revenue_df.empty:
            revenue_by_type = {}
            for _, row in revenue_df.iterrows():
                txn_type = row['transaction_type']
                count = int(row['transaction_count'])
                total = float(row['total_revenue'])
                avg = float(row['avg_revenue'])
                
                revenue_by_type[txn_type] = {
                    'count': count,
                    'total_revenue': total,
                    'avg_revenue': avg
                }
            
            results['revenue_check'] = revenue_by_type
            
            # Calculate gross revenue (Shipment only)
            if 'Shipment' in revenue_by_type:
                gross_revenue = revenue_by_type['Shipment']['total_revenue']
                results['revenue_check']['gross_revenue'] = gross_revenue
            else:
                results['warnings'].append("⚠️  No 'Shipment' transactions found")
        
        # 6. Generate recommendations
        if results['issues']:
            if any('DUPLICATE' in issue for issue in results['issues']):
                results['recommendations'].append(
                    "1. CLEAR database: POST /api/upload/reset-database"
                )
                results['recommendations'].append(
                    "2. RE-UPLOAD CSV files (ensure no duplicates in source files)"
                )
                results['recommendations'].append(
                    "3. Verify counts match original after re-upload"
                )
            
            if any('empty order_ids' in issue.lower() for issue in results['issues']):
                results['recommendations'].append(
                    "4. Check source CSV files for missing 'Order Id' values"
                )
                results['recommendations'].append(
                    "5. Ensure 'Order Id' column is properly mapped during upload"
                )
        else:
            results['recommendations'].append("✅ Data integrity verified - no issues found")
        
    except Exception as e:
        results['issues'].append(f"❌ Error during verification: {str(e)}")
        import traceback
        results['error'] = traceback.format_exc()
    
    return results


@router.get("/check-schema")
async def check_schema():
    """
    Check current database schema and report column naming issues
    
    Returns analysis showing:
    - Whether old column names exist ("Invoice Amount", etc.)
    - Whether standardized names exist ("revenue_amount", etc.)
    - Whether lineage columns exist (source_file, ingestion_id, loaded_at)
    - Recommendation for action
    """
    analysis = check_database_schema('sales')
    
    return analysis
