"""
Chart/Diagnostics API Endpoints with tenant isolation
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
import logging

from core.database import execute_query, table_exists, get_connection
from api.deps.auth_deps import get_current_user
from models.user import UserInDB
from utils.tenant_filter import get_tenant_filter_sql

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/refunds-by-source")
async def get_refunds_by_source(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Diagnostic endpoint to see refunds breakdown by source file
    
    Helps identify if multiple CSV files are contributing to refund totals
    
    TENANT ISOLATION: Results are filtered by user's tenant_id.
    """
    if not table_exists('sales'):
        return {"error": "Sales table does not exist"}
    
    # TENANT ISOLATION: Get tenant filter
    tenant_id = current_user.tenant_id or str(current_user.id)
    tenant_filter = get_tenant_filter_sql(tenant_id)
    
    try:
        # Find column names
        column_info = execute_query("DESCRIBE sales")
        existing_columns = column_info['column_name'].tolist()
        
        # Find date column
        date_col = None
        for col in ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']:
            if col in existing_columns:
                date_col = col
                break
        
        # Find transaction type column
        txn_col = None
        for col in ['Transaction Type', 'transaction_type']:
            if col in existing_columns:
                txn_col = col
                break
        
        # Find amount column
        amount_col = None
        for col in ['Invoice Amount', 'revenue_amount', 'Invoice Amount']:
            if col in existing_columns:
                amount_col = col
                break
        
        if not date_col or not txn_col or not amount_col:
            return {"error": "Required columns not found"}
        
        # Build date filter
        date_filter = ""
        if start_date and end_date:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper():
                date_filter = f"AND CAST(\"{date_col}\" AS DATE) >= '{start_date}' AND CAST(\"{date_col}\" AS DATE) <= '{end_date}'"
            else:
                date_filter = f"AND \"{date_col}\" >= '{start_date}' AND \"{date_col}\" <= '{end_date}'"
        
        # Query refunds by source file with tenant isolation
        sql = f"""
        SELECT 
            COALESCE(source_file, 'Unknown') as source_file,
            COUNT(*) as refund_count,
            SUM(ABS("{amount_col}")) as total_refunds,
            MIN(CAST("{date_col}" AS DATE)) as min_date,
            MAX(CAST("{date_col}" AS DATE)) as max_date
        FROM sales
        WHERE {tenant_filter} AND "{txn_col}" = 'Refund'
        {date_filter}
        GROUP BY source_file
        ORDER BY total_refunds DESC
        """
        
        result_df = execute_query(sql)
        
        return {
            "data": result_df.to_dict('records'),
            "total_refunds": float(result_df['total_refunds'].sum()) if not result_df.empty else 0,
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting refunds by source: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/july-refunds-breakdown")
async def get_july_refunds_breakdown(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Detailed breakdown of July 2025 refunds by source file
    Helps diagnose why database total (414,186) doesn't match CSV total (332,996)
    
    TENANT ISOLATION: Results are filtered by user's tenant_id.
    """
    if not table_exists('sales'):
        return {"error": "Sales table does not exist"}
    
    # TENANT ISOLATION: Get tenant filter
    tenant_id = current_user.tenant_id or str(current_user.id)
    tenant_filter = get_tenant_filter_sql(tenant_id)
    
    try:
        # Find column names
        column_info = execute_query("DESCRIBE sales")
        existing_columns = column_info['column_name'].tolist()
        
        # Find date column
        date_col = None
        for col in ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']:
            if col in existing_columns:
                date_col = col
                break
        
        # Find transaction type column
        txn_col = None
        for col in ['Transaction Type', 'transaction_type']:
            if col in existing_columns:
                txn_col = col
                break
        
        # Find amount column
        amount_col = None
        for col in ['Invoice Amount', 'revenue_amount', 'Invoice Amount']:
            if col in existing_columns:
                amount_col = col
                break
        
        if not date_col or not txn_col or not amount_col:
            return {"error": "Required columns not found"}
        
        # Check column type for date
        col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
        date_cast = f'CAST("{date_col}" AS DATE)' if 'VARCHAR' in str(col_type).upper() else f'"{date_col}"'
        
        # 1. Total refunds for July 2025 (with tenant isolation)
        total_sql = f"""
        SELECT SUM(ABS("{amount_col}")) as total_refunds, COUNT(*) as count
        FROM sales
        WHERE {tenant_filter} AND "{txn_col}" = 'Refund'
        AND {date_cast} >= '2025-07-01'
        AND {date_cast} <= '2025-07-31'
        """
        total_df = execute_query(total_sql)
        
        # 2. Breakdown by source file (with tenant isolation)
        breakdown_sql = f"""
        SELECT 
            COALESCE(source_file, 'Unknown') as source_file,
            COUNT(*) as refund_count,
            SUM(ABS("{amount_col}")) as total_refunds,
            MIN({date_cast}) as min_date,
            MAX({date_cast}) as max_date
        FROM sales
        WHERE {tenant_filter} AND "{txn_col}" = 'Refund'
        AND {date_cast} >= '2025-07-01'
        AND {date_cast} <= '2025-07-31'
        GROUP BY source_file
        ORDER BY total_refunds DESC
        """
        breakdown_df = execute_query(breakdown_sql)
        
        # 3. Check for duplicates (same invoice number appearing multiple times)
        invoice_col = None
        for col in ['Invoice Number', 'Invoice Number', 'order_id', 'Invoice Number']:
            if col in existing_columns:
                invoice_col = col
                break
        
        duplicates_info = None
        if invoice_col:
            dup_sql = f"""
            SELECT 
                "{invoice_col}",
                COUNT(*) as duplicate_count,
                STRING_AGG(DISTINCT source_file, ', ') as source_files,
                SUM(ABS("{amount_col}")) as total_amount
            FROM sales
            WHERE {tenant_filter} AND "{txn_col}" = 'Refund'
            AND {date_cast} >= '2025-07-01'
            AND {date_cast} <= '2025-07-31'
            GROUP BY "{invoice_col}"
            HAVING COUNT(*) > 1
            ORDER BY duplicate_count DESC
            LIMIT 20
            """
            dup_df = execute_query(dup_sql)
            if not dup_df.empty:
                duplicates_info = {
                    "found": True,
                    "count": len(dup_df),
                    "total_duplicate_rows": int(dup_df['duplicate_count'].sum()) - len(dup_df),
                    "examples": dup_df.head(10).to_dict('records')
                }
        
        return {
            "july_2025_total": float(total_df.iloc[0]['total_refunds']) if not total_df.empty else 0,
            "july_2025_count": int(total_df.iloc[0]['count']) if not total_df.empty else 0,
            "expected_csv_total": 332996.41,
            "difference": float(total_df.iloc[0]['total_refunds']) - 332996.41 if not total_df.empty else 0,
            "breakdown_by_source": breakdown_df.to_dict('records') if not breakdown_df.empty else [],
            "duplicates": duplicates_info if duplicates_info else {"found": False, "message": "No duplicate invoice numbers found"},
            "diagnosis": _diagnose_discrepancy(
                float(total_df.iloc[0]['total_refunds']) if not total_df.empty else 0,
                332996.41,
                breakdown_df if not breakdown_df.empty else None
            )
        }
    
    except Exception as e:
        logger.error(f"Error in July refunds breakdown: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def _diagnose_discrepancy(db_total: float, csv_total: float, breakdown_df) -> str:
    """Provide diagnosis of why database total doesn't match CSV"""
    difference = db_total - csv_total
    
    if difference < 100:
        return "Values are close - likely rounding differences or minor data variations"
    elif breakdown_df is not None and len(breakdown_df) > 1:
        return f"Multiple source files detected. Database includes refunds from {len(breakdown_df)} different CSV files. CSV shows only one file's data."
    elif breakdown_df is not None and len(breakdown_df) == 1:
        source = breakdown_df.iloc[0]['source_file']
        source_total = breakdown_df.iloc[0]['total_refunds']
        if abs(source_total - db_total) < 1:
            return f"Single source file ({source}) matches database total. CSV filter might be excluding some rows or showing filtered subset."
        else:
            return f"Single source file ({source}) but totals don't match. Possible duplicate data or data transformation differences."
    else:
        return f"Difference of ₹{difference:,.2f} detected. Check for: 1) Duplicate records, 2) Data from multiple source files, 3) CSV filter excluding rows"
