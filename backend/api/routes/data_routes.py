"""
Data Routes - Raw transaction data endpoints with tenant isolation
"""
from fastapi import APIRouter, Query, HTTPException, UploadFile, File, Depends, Body
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from services.data_service import get_transactions, get_unique_skus, get_data_statistics, export_transactions_csv
from services.upload_service import process_csv_upload
from utils.validators import (
    validate_date_range,
    validate_date,
    validate_sku,
    validate_transaction_type,
    validate_page,
    validate_limit,
)
from utils.sanitizers import (
    sanitize_string,
    sanitize_date_string,
    sanitize_sku,
    sanitize_transaction_type,
    sanitize_integer,
)
from utils.error_handler import format_error_response, log_error
from api.deps.auth_deps import get_current_user
from models.user import UserInDB

router = APIRouter()


@router.get("/transactions")
async def get_transactions_endpoint(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(50, ge=1, le=100, description="Number of rows per page"),
    date_from: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    sku: Optional[str] = Query(None, description="SKU filter (exact match)"),
    transaction_type: Optional[str] = Query(None, description="Transaction type filter (Shipment/Refund/Cancel)"),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get paginated transaction data with optional filters
    
    Filters:
        - date_from: Filter transactions from this date onwards
        - date_to: Filter transactions up to this date
        - sku: Filter by specific SKU (exact match)
        - transaction_type: Filter by transaction type (Shipment, Refund, Cancel, etc.)
    
    Returns:
        {
            "data": [
                {
                    "order_id": "ORD123",
                    "sku": "GP10_OM2P_STARTRC",
                    "transaction_type": "Shipment",
                    "amount": 1500.00,
                    "date": "2025-07-15",
                    "quantity": 2
                },
                ...
            ],
            "total": 15507,
            "page": 1,
            "total_pages": 311
        }
    """
    try:
        # Sanitize inputs
        page = sanitize_integer(page, default=1)
        limit = sanitize_integer(limit, default=50)
        date_from = sanitize_date_string(date_from)
        date_to = sanitize_date_string(date_to)
        sku = sanitize_sku(sku)
        transaction_type = sanitize_transaction_type(transaction_type)
        
        # Validate inputs
        validate_page(page)
        validate_limit(limit)
        
        if date_from and date_to:
            validate_date_range(date_from, date_to)
        elif date_from:
            validate_date(date_from, "date_from")
        elif date_to:
            validate_date(date_to, "date_to")
        
        if sku:
            validate_sku(sku)
        
        if transaction_type:
            validate_transaction_type(transaction_type)
        
        # TENANT ISOLATION: Pass user's tenant_id
        tenant_id = current_user.tenant_id or str(current_user.id)
        
        result = get_transactions(
            page=page,
            limit=limit,
            date_from=date_from,
            date_to=date_to,
            sku=sku,
            transaction_type=transaction_type,
            tenant_id=tenant_id,
        )
        return result
    except ValueError as e:
        log_error(e, 'get_transactions', {
            'page': page, 'limit': limit, 'date_from': date_from,
            'date_to': date_to, 'sku': sku, 'transaction_type': transaction_type
        })
        error_response = format_error_response(e, status_code=400, user_message=str(e))
        raise HTTPException(status_code=400, detail=error_response)
    except Exception as e:
        log_error(e, 'get_transactions', {
            'page': page, 'limit': limit, 'date_from': date_from,
            'date_to': date_to, 'sku': sku, 'transaction_type': transaction_type
        })
        error_response = format_error_response(e, status_code=500, user_message="Failed to fetch transactions. Please try again later.")
        raise HTTPException(status_code=500, detail=error_response)


@router.get("/skus")
async def get_skus_endpoint(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get list of unique SKUs from sales table (tenant-isolated).
    
    Returns:
        {
            "skus": ["GP10_OM2P_STARTRC", "GP947-L_TLSnk", ...]
        }
    """
    try:
        # TENANT ISOLATION: Pass user's tenant_id
        tenant_id = current_user.tenant_id or str(current_user.id)
        result = get_unique_skus(tenant_id=tenant_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats_endpoint(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get data statistics: total records, date range, unique SKUs (tenant-isolated).
    
    Returns:
        {
            "total_records": 15507,
            "date_range": {"start": "2025-07-01", "end": "2025-09-30"},
            "unique_skus": 2847
        }
    """
    try:
        # TENANT ISOLATION: Pass user's tenant_id
        tenant_id = current_user.tenant_id or str(current_user.id)
        result = get_data_statistics(tenant_id=tenant_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_data_summary_endpoint(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Lightweight endpoint to check if user has any data (tenant-isolated).
    
    Used by frontend to determine whether to show empty states or actual content.
    
    Returns:
        {
            "has_data": true/false,
            "row_count": 15507
        }
    """
    try:
        from core.database import table_exists, execute_query
        from utils.tenant_filter import get_tenant_filter_sql
        
        # TENANT ISOLATION: Check data for this user only
        tenant_id = current_user.tenant_id or str(current_user.id)
        
        if not table_exists('sales'):
            return {
                "has_data": False,
                "row_count": 0
            }
        
        tenant_filter = get_tenant_filter_sql(tenant_id)
        count_result = execute_query(f"SELECT COUNT(*) as count FROM sales WHERE {tenant_filter}")
        row_count = int(count_result.iloc[0]['count']) if not count_result.empty else 0
        
        return {
            "has_data": row_count > 0,
            "row_count": row_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export")
async def export_transactions_endpoint(
    format: str = Query("csv", description="Export format: csv or parquet"),
    date_from: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    sku: Optional[str] = Query(None, description="SKU filter (exact match)"),
    transaction_type: Optional[str] = Query(None, description="Transaction type filter"),
    ingestion_id: Optional[str] = Query(None, description="Filter by ingestion ID"),
    source_file: Optional[str] = Query(None, description="Filter by source file name"),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Export filtered transactions as CSV or Parquet file (tenant-isolated).
    
    Supports streaming for large exports. Max 100,000 rows without filters.
    If no filters provided and data exceeds limit, returns 400 error.
    
    Args:
        format: Export format - "csv" (default) or "parquet"
        date_from: Start date filter (YYYY-MM-DD)
        date_to: End date filter (YYYY-MM-DD)
        sku: SKU filter (exact match)
        transaction_type: Transaction type filter
        ingestion_id: Filter by ingestion ID
        source_file: Filter by source file name
        current_user: Authenticated user (from JWT token)
    
    Returns:
        StreamingResponse with CSV or Parquet file download
    """
    from services.data_service import export_transactions_csv, export_transactions_parquet, export_transactions_csv_streaming
    
    # Validate format
    if format not in ["csv", "parquet"]:
        raise HTTPException(status_code=400, detail="Format must be 'csv' or 'parquet'")
    
    # TENANT ISOLATION: Pass user's tenant_id
    tenant_id = current_user.tenant_id or str(current_user.id)
    
    try:
        # Check if filters are provided (for max rows validation)
        has_filters = any([date_from, date_to, sku, transaction_type, ingestion_id, source_file])
        
        if format == "parquet":
            # Parquet export
            parquet_path = export_transactions_parquet(
                date_from=date_from,
                date_to=date_to,
                sku=sku,
                transaction_type=transaction_type,
                ingestion_id=ingestion_id,
                source_file=source_file,
                tenant_id=tenant_id,
            )
            
            filename = f"transactions_export_{datetime.now().strftime('%Y-%m-%d')}.parquet"
            
            def cleanup_temp_file():
                """Cleanup temp file after streaming"""
                import os
                try:
                    if os.path.exists(parquet_path):
                        os.remove(parquet_path)
                except Exception:
                    pass
            
            from fastapi.responses import FileResponse
            return FileResponse(
                parquet_path,
                media_type="application/octet-stream",
                filename=filename,
                background=cleanup_temp_file,
            )
        else:
            # CSV export (streaming)
            filename = f"transactions_export_{datetime.now().strftime('%Y-%m-%d')}.csv"
            
            # Use streaming export for large files
            return StreamingResponse(
                export_transactions_csv_streaming(
                    date_from=date_from,
                    date_to=date_to,
                    sku=sku,
                    transaction_type=transaction_type,
                    ingestion_id=ingestion_id,
                    source_file=source_file,
                    tenant_id=tenant_id,
                    max_rows=100000,
                    require_filters=not has_filters,
                ),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error exporting transactions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_csv_endpoint(
    file: UploadFile = File(..., description="CSV file to upload"),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Upload CSV file to database with tenant isolation.
    
    Requires authentication. All uploaded data is scoped to the user's tenant_id.
    
    Returns:
        {
            "success": true,
            "message": "Upload successful",
            "rows_inserted": 1500,
            "filename": "example.csv"
        }
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
    
    # TENANT ISOLATION: Pass user's tenant_id to upload service
    # All uploaded data will be tagged with this tenant_id
    tenant_id = current_user.tenant_id
    if not tenant_id:
        # Fallback: use user ID as tenant_id
        tenant_id = str(current_user.id)
    
    # Process upload using existing upload service with tenant isolation
    result = process_csv_upload(file_content, file.filename, tenant_id=tenant_id)
    
    if not result.get('success'):
        # Handle duplicate upload (409)
        if result.get('error') == 'DUPLICATE_UPLOAD' or result.get('status_code') == 409:
            raise HTTPException(
                status_code=409,
                detail={
                    'error': 'DUPLICATE_UPLOAD',
                    'message': result.get('message', 'This file has already been uploaded'),
                    'existing_ingestion_id': result.get('existing_ingestion_id'),
                    'existing_filename': result.get('existing_filename'),
                    'existing_uploaded_at': result.get('existing_uploaded_at'),
                    'existing_rows': result.get('existing_rows', 0),
                }
            )
        # Handle required columns missing (400)
        elif result.get('error') == 'REQUIRED_COLUMNS_MISSING' or result.get('status_code') == 400:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'REQUIRED_COLUMNS_MISSING',
                    'message': result.get('message', 'Required columns are missing'),
                    'missing_columns': result.get('missing_columns', []),
                    'expected_schema': result.get('expected_schema', {}),
                    'csv_columns': result.get('csv_columns', []),
                    'detected_mapping': result.get('detected_mapping', {}),
                }
            )
        else:
            raise HTTPException(
                status_code=result.get('status_code', 500),
                detail=result.get('error', 'Upload processing failed')
            )
    
    return {
        "success": True,
        "message": f"Successfully uploaded {result.get('rows_inserted', result.get('rows_uploaded', 0))} rows",
        "rows_inserted": result.get('rows_inserted', result.get('rows_uploaded', 0)),
        "filename": result.get('filename'),
        "ingestion_id": result.get('ingestion_id'),
        "tenant_id": tenant_id,
    }


class ResetDataRequest(BaseModel):
    """Request model for reset tenant data endpoint"""
    confirm: str  # Must be "DELETE" to confirm


@router.post("/reset")
async def reset_tenant_data(
    request: ResetDataRequest = Body(...),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Reset all data for the current tenant (self-serve).
    
    TENANT ISOLATION: Only deletes data belonging to the current user's tenant.
    This includes:
    - All sales rows with tenant_id = current_user.tenant_id
    - Tenant's ChromaDB collection (rag_{tenant_id})
    - Ingestion logs for this tenant's files
    
    Requires confirmation: request body must include {"confirm": "DELETE"}
    
    Args:
        request: Confirmation request body
        current_user: Authenticated user (from JWT token)
    
    Returns:
        {
            "success": bool,
            "deleted_rows": int,
            "message": str
        }
    """
    from core.database import get_connection, table_exists
    from utils.tenant_filter import get_tenant_filter_sql
    from services.embedding_service import get_chroma_client
    
    # Require confirmation
    if request.confirm != "DELETE":
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Send {'confirm': 'DELETE'} to reset all your data."
        )
    
    # TENANT ISOLATION: Get tenant_id from authenticated user
    tenant_id = current_user.tenant_id or str(current_user.id)
    
    try:
        conn = get_connection()
        tenant_filter = get_tenant_filter_sql(tenant_id)
        
        deleted_rows = 0
        
        # 1. Delete all sales rows for this tenant
        if table_exists('sales'):
            count_sql = f"SELECT COUNT(*) as count FROM sales WHERE {tenant_filter}"
            count_df = conn.execute(count_sql).fetchdf()
            if not count_df.empty:
                deleted_rows = int(count_df.iloc[0]['count'])
            
            if deleted_rows > 0:
                delete_sql = f"DELETE FROM sales WHERE {tenant_filter}"
                conn.execute(delete_sql)
        
        # 2. Delete ingestion logs for this tenant (tenant-isolated)
        if table_exists('ingestion_log'):
            delete_log_sql = f"DELETE FROM ingestion_log WHERE {tenant_filter}"
            conn.execute(delete_log_sql)
        
        # 3. Delete tenant's ChromaDB collection
        try:
            chroma_client = get_chroma_client()
            collection_name = f"rag_{tenant_id}"
            try:
                collection = chroma_client.get_collection(name=collection_name)
                collection.delete()  # Delete all documents in collection
                # Note: ChromaDB doesn't have a direct "delete collection" API, but deleting all docs effectively clears it
            except Exception:
                # Collection doesn't exist, which is fine
                pass
        except Exception as e:
            # ChromaDB deletion failed, but continue with other cleanup
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to delete ChromaDB collection for tenant {tenant_id}: {e}")
        
        return {
            "success": True,
            "deleted_rows": deleted_rows,
            "message": f"Successfully reset all data for your account. Deleted {deleted_rows} rows."
        }
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error resetting tenant data: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting data: {str(e)}")


@router.get("/ingestions")
async def get_ingestions_endpoint(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get list of ingestions (uploaded files) for the current tenant.
    
    TENANT ISOLATION: Only returns ingestions belonging to the current user's tenant.
    
    Returns:
        {
            "ingestions": [
                {
                    "ingestion_id": str,
                    "filename": str,
                    "uploaded_at": str,
                    "rows_inserted": int,
                    "file_size": int,
                    "date_range_start": str | null,
                    "date_range_end": str | null,
                    "validation_status": str
                }
            ]
        }
    """
    from core.database import get_connection, table_exists
    from utils.tenant_filter import get_tenant_filter_sql
    
    # TENANT ISOLATION: Get tenant_id from authenticated user
    tenant_id = current_user.tenant_id or str(current_user.id)
    
    if not table_exists('ingestion_log'):
        return {"ingestions": []}
    
    try:
        conn = get_connection()
        tenant_filter = get_tenant_filter_sql(tenant_id)
        
        sql = f"""
        SELECT
            ingestion_id,
            filename,
            uploaded_at,
            rows_inserted,
            file_size,
            date_range_start,
            date_range_end,
            validation_status,
            processing_time_seconds
        FROM ingestion_log
        WHERE {tenant_filter}
        ORDER BY uploaded_at DESC
        """
        
        result_df = conn.execute(sql).fetchdf()
        
        if result_df.empty:
            return {"ingestions": []}
        
        # Convert to list of dicts
        ingestions = result_df.to_dict('records')
        
        # Format dates as strings
        for ingestion in ingestions:
            if 'uploaded_at' in ingestion and pd.notna(ingestion['uploaded_at']):
                ingestion['uploaded_at'] = str(ingestion['uploaded_at'])
            if 'date_range_start' in ingestion and pd.notna(ingestion['date_range_start']):
                ingestion['date_range_start'] = str(ingestion['date_range_start'])
            if 'date_range_end' in ingestion and pd.notna(ingestion['date_range_end']):
                ingestion['date_range_end'] = str(ingestion['date_range_end'])
        
        return {"ingestions": ingestions}
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error fetching ingestions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

