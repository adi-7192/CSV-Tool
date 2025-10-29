"""
Chart Data Endpoints - Serve data formatted for frontend charts
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from core.database import execute_query, table_exists

router = APIRouter()


@router.get("/regional-distribution")
async def regional_distribution(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Get revenue distribution by region (for pie/donut charts)
    """
    try:
        if not table_exists('sales'):
            return {"data": [], "count": 0}
        
        # Find column names dynamically
        column_info = execute_query("DESCRIBE sales")
        
        # Find region/city column
        region_cols = ['Ship To City', 'Region', 'region', 'city', 'City']
        region_col = None
        for col in region_cols:
            if col in column_info['column_name'].values:
                region_col = col
                break
        
        # Find revenue column
        revenue_cols = ['revenue_calc', 'revenue_in_inr', 'Revenue Amount']
        revenue_col = None
        for col in revenue_cols:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        # Find transaction type column
        txn_cols = ['Transaction Type', 'transaction_type']
        txn_col = None
        for col in txn_cols:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        # Find date column
        date_cols = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        date_col = None
        for col in date_cols:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        if not region_col or not revenue_col:
            return {"data": [], "count": 0}
        
        # Build date filter
        date_filter = ""
        if start_date and end_date and date_col:
            date_filter = f'AND "{date_col}" >= \'{start_date}\' AND "{date_col}" <= \'{end_date}\''
        
        # Build transaction type filter
        txn_filter = ""
        if txn_col:
            txn_filter = f'AND "{txn_col}" = \'Shipment\''
        
        sql = f"""
        SELECT 
            "{region_col}" as region,
            SUM({revenue_col}) as revenue,
            COUNT(DISTINCT "Invoice Number") as orders
        FROM sales
        WHERE 1=1 {txn_filter} {date_filter}
        GROUP BY "{region_col}"
        ORDER BY revenue DESC
        LIMIT 10
        """
        
        result_df = execute_query(sql)
        
        return {
            "data": result_df.to_dict('records'),
            "count": len(result_df),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transaction-status")
async def transaction_status_distribution(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Get distribution of transaction types (Shipment/Refund/Cancel)
    """
    try:
        if not table_exists('sales'):
            return {"data": []}
        
        # Find column names dynamically
        column_info = execute_query("DESCRIBE sales")
        
        # Find transaction type column
        txn_cols = ['Transaction Type', 'transaction_type']
        txn_col = None
        for col in txn_cols:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        # Find revenue column
        revenue_cols = ['revenue_calc', 'revenue_in_inr', 'Revenue Amount']
        revenue_col = None
        for col in revenue_cols:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        # Find date column
        date_cols = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        date_col = None
        for col in date_cols:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        if not txn_col:
            return {"data": []}
        
        # Build date filter
        date_filter = ""
        if start_date and end_date and date_col:
            date_filter = f'AND "{date_col}" >= \'{start_date}\' AND "{date_col}" <= \'{end_date}\''
        
        sql = f"""
        SELECT 
            "{txn_col}" as transaction_type,
            COUNT(*) as count,
            SUM({revenue_col}) as total_amount
        FROM sales
        WHERE 1=1 {date_filter}
        GROUP BY "{txn_col}"
        ORDER BY count DESC
        """
        
        result_df = execute_query(sql)
        
        return {
            "data": result_df.to_dict('records'),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
