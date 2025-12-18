"""
Data Routes - Raw transaction data endpoints with tenant isolation
"""
from fastapi import APIRouter, Query, HTTPException, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from typing import Optional
from datetime import datetime
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
    date_from: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    sku: Optional[str] = Query(None, description="SKU filter (exact match)"),
    transaction_type: Optional[str] = Query(None, description="Transaction type filter"),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Export filtered transactions as CSV file (tenant-isolated).
    
    Parameters same as /transactions endpoint.
    Returns CSV file download.
    """
    try:
        # TENANT ISOLATION: Pass user's tenant_id
        tenant_id = current_user.tenant_id or str(current_user.id)
        csv_buffer = export_transactions_csv(
            date_from=date_from,
            date_to=date_to,
            sku=sku,
            transaction_type=transaction_type,
            tenant_id=tenant_id,
        )
        
        # Generate filename with current date
        filename = f"transactions_export_{datetime.now().strftime('%Y-%m-%d')}.csv"
        
        return StreamingResponse(
            csv_buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
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
        raise HTTPException(
            status_code=500,
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

