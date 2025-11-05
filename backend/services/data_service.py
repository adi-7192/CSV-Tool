"""
Data Service - Handle raw transaction data queries
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from io import BytesIO
from core.database import execute_query, table_exists
import logging

logger = logging.getLogger(__name__)


def get_transactions(
    page: int = 1,
    limit: int = 50,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sku: Optional[str] = None,
    transaction_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get paginated transaction data from sales table
    
    Args:
        page: Page number (1-indexed)
        limit: Number of rows per page
    
    Returns:
        Dictionary with data, total count, page info
    """
    if not table_exists('sales'):
        return {
            "data": [],
            "total": 0,
            "page": page,
            "total_pages": 0,
        }
    
    try:
        # Detect column names dynamically
        column_info = execute_query("DESCRIBE sales")
        
        # Find column names
        order_id_col = None
        sku_col = None
        txn_col = None
        revenue_col = None
        date_col = None
        quantity_col = None
        
        order_id_columns = ['order_id', 'Invoice Number', 'Order ID']
        sku_columns = ['sku', 'Sku', 'SKU']
        txn_columns = ['transaction_type', 'Transaction Type']
        revenue_columns = ['revenue_amount', 'Invoice Amount', 'revenue_in_inr']
        date_columns = ['order_date', 'Invoice Date', 'invoice_date', 'Order Date']
        quantity_columns = ['quantity', 'Quantity', 'units_sold']
        
        for col in order_id_columns:
            if col in column_info['column_name'].values:
                order_id_col = col
                break
        
        for col in sku_columns:
            if col in column_info['column_name'].values:
                sku_col = col
                break
        
        for col in txn_columns:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        for col in revenue_columns:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        for col in quantity_columns:
            if col in column_info['column_name'].values:
                quantity_col = col
                break
        
        if not order_id_col or not sku_col or not revenue_col:
            logger.error("Required columns not found in sales table")
            return {
                "data": [],
                "total": 0,
                "page": page,
                "total_pages": 0,
            }
        
        # Calculate offset
        offset = (page - 1) * limit
        
        # Build SELECT columns
        select_cols = [
            f'"{order_id_col}" as order_id',
            f'"{sku_col}" as sku',
        ]
        
        if txn_col:
            select_cols.append(f'"{txn_col}" as transaction_type')
        else:
            select_cols.append("'Shipment' as transaction_type")
        
        select_cols.append(f'ABS({revenue_col}) as amount')
        
        if date_col:
            # Check if date column needs casting
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                select_cols.append(f'CAST("{date_col}" AS DATE) as date')
            else:
                select_cols.append(f'"{date_col}" as date')
        else:
            select_cols.append("NULL as date")
        
        if quantity_col:
            select_cols.append(f'COALESCE({quantity_col}, 0) as quantity')
        else:
            select_cols.append('0 as quantity')
        
        # Build query
        select_clause = ', '.join(select_cols)
        
        # Build WHERE clause with filters
        where_conditions = []
        
        # Date filter
        if date_from and date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                where_conditions.append(f'CAST("{date_col}" AS DATE) >= \'{date_from}\'')
            else:
                where_conditions.append(f'"{date_col}" >= \'{date_from}\'')
        
        if date_to and date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                where_conditions.append(f'CAST("{date_col}" AS DATE) <= \'{date_to}\'')
            else:
                where_conditions.append(f'"{date_col}" <= \'{date_to}\'')
        
        # SKU filter
        if sku:
            where_conditions.append(f'"{sku_col}" = \'{sku}\'')
        
        # Transaction type filter
        if transaction_type and txn_col:
            where_conditions.append(f'"{txn_col}" = \'{transaction_type}\'')
        
        where_clause = ' AND '.join(where_conditions) if where_conditions else '1=1'
        
        # Get total count with filters
        count_sql = f"SELECT COUNT(*) as total FROM sales WHERE {where_clause}"
        count_df = execute_query(count_sql)
        total = int(count_df['total'].iloc[0]) if not count_df.empty else 0
        
        # Get paginated data
        sql = f"""
        SELECT {select_clause}
        FROM sales
        WHERE {where_clause}
        ORDER BY {date_col if date_col else order_id_col} DESC
        LIMIT {limit} OFFSET {offset}
        """
        
        df = execute_query(sql)
        
        # Convert DataFrame to list of dicts
        data = []
        for _, row in df.iterrows():
            data.append({
                "order_id": str(row.get('order_id', '')),
                "sku": str(row.get('sku', '')),
                "transaction_type": str(row.get('transaction_type', '')),
                "amount": float(row.get('amount', 0)),
                "date": str(row.get('date', '')) if row.get('date') is not None else '',
                "quantity": int(row.get('quantity', 0)),
            })
        
        # Calculate total pages
        total_pages = (total + limit - 1) // limit if total > 0 else 0
        
        return {
            "data": data,
            "total": total,
            "page": page,
            "total_pages": total_pages,
        }
        
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "data": [],
            "total": 0,
            "page": page,
            "total_pages": 0,
        }


def get_unique_skus() -> Dict[str, Any]:
    """
    Get list of unique SKU values from sales table
    
    Returns:
        Dictionary with list of SKUs
    """
    if not table_exists('sales'):
        return {
            "skus": [],
        }
    
    try:
        # Detect SKU column name
        column_info = execute_query("DESCRIBE sales")
        sku_columns = ['sku', 'Sku', 'SKU']
        sku_col = None
        
        for col in sku_columns:
            if col in column_info['column_name'].values:
                sku_col = col
                break
        
        if not sku_col:
            logger.error("SKU column not found in sales table")
            return {
                "skus": [],
            }
        
        # Query distinct SKUs
        sql = f"""
        SELECT DISTINCT "{sku_col}" as sku
        FROM sales
        WHERE "{sku_col}" IS NOT NULL AND TRIM("{sku_col}") != ''
        ORDER BY "{sku_col}" ASC
        """
        
        df = execute_query(sql)
        
        # Convert to list
        skus = [str(row['sku']) for _, row in df.iterrows() if row['sku']]
        
        return {
            "skus": skus,
        }
        
    except Exception as e:
        logger.error(f"Error getting unique SKUs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "skus": [],
        }


def get_data_statistics() -> Dict[str, Any]:
    """
    Get data statistics: total records, date range, unique SKUs
    
    Returns:
        Dictionary with statistics
    """
    if not table_exists('sales'):
        return {
            "total_records": 0,
            "date_range": {"start": None, "end": None},
            "unique_skus": 0,
        }
    
    try:
        # Detect column names
        column_info = execute_query("DESCRIBE sales")
        
        date_columns = ['order_date', 'Invoice Date', 'invoice_date', 'Order Date']
        sku_columns = ['sku', 'Sku', 'SKU']
        
        date_col = None
        sku_col = None
        
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        for col in sku_columns:
            if col in column_info['column_name'].values:
                sku_col = col
                break
        
        # Get total records
        count_sql = "SELECT COUNT(*) as total FROM sales"
        count_df = execute_query(count_sql)
        total_records = int(count_df['total'].iloc[0]) if not count_df.empty else 0
        
        # Get date range
        date_start = None
        date_end = None
        if date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                range_sql = f"SELECT MIN(CAST(\"{date_col}\" AS DATE)) as min_date, MAX(CAST(\"{date_col}\" AS DATE)) as max_date FROM sales WHERE \"{date_col}\" IS NOT NULL"
            else:
                range_sql = f"SELECT MIN(\"{date_col}\") as min_date, MAX(\"{date_col}\") as max_date FROM sales WHERE \"{date_col}\" IS NOT NULL"
            
            range_df = execute_query(range_sql)
            if not range_df.empty:
                min_date = range_df['min_date'].iloc[0]
                max_date = range_df['max_date'].iloc[0]
                if min_date is not None:
                    date_start = str(min_date)
                if max_date is not None:
                    date_end = str(max_date)
        
        # Get unique SKUs count
        unique_skus = 0
        if sku_col:
            sku_sql = f"SELECT COUNT(DISTINCT \"{sku_col}\") as count FROM sales WHERE \"{sku_col}\" IS NOT NULL AND TRIM(\"{sku_col}\") != ''"
            sku_df = execute_query(sku_sql)
            if not sku_df.empty:
                unique_skus = int(sku_df['count'].iloc[0])
        
        return {
            "total_records": total_records,
            "date_range": {
                "start": date_start,
                "end": date_end,
            },
            "unique_skus": unique_skus,
        }
        
    except Exception as e:
        logger.error(f"Error getting data statistics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "total_records": 0,
            "date_range": {"start": None, "end": None},
            "unique_skus": 0,
        }


def export_transactions_csv(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sku: Optional[str] = None,
    transaction_type: Optional[str] = None,
) -> BytesIO:
    """
    Export filtered transactions to CSV
    
    Args:
        date_from: Start date filter
        date_to: End date filter
        sku: SKU filter
        transaction_type: Transaction type filter
    
    Returns:
        BytesIO object with CSV content
    """
    if not table_exists('sales'):
        # Return empty CSV
        df = pd.DataFrame(columns=['order_id', 'sku', 'transaction_type', 'amount', 'date', 'quantity'])
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
    
    try:
        # Detect column names dynamically (same logic as get_transactions)
        column_info = execute_query("DESCRIBE sales")
        
        order_id_col = None
        sku_col = None
        txn_col = None
        revenue_col = None
        date_col = None
        quantity_col = None
        
        order_id_columns = ['order_id', 'Invoice Number', 'Order ID']
        sku_columns = ['sku', 'Sku', 'SKU']
        txn_columns = ['transaction_type', 'Transaction Type']
        revenue_columns = ['revenue_amount', 'Invoice Amount', 'revenue_in_inr']
        date_columns = ['order_date', 'Invoice Date', 'invoice_date', 'Order Date']
        quantity_columns = ['quantity', 'Quantity', 'units_sold']
        
        for col in order_id_columns:
            if col in column_info['column_name'].values:
                order_id_col = col
                break
        
        for col in sku_columns:
            if col in column_info['column_name'].values:
                sku_col = col
                break
        
        for col in txn_columns:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        for col in revenue_columns:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        for col in quantity_columns:
            if col in column_info['column_name'].values:
                quantity_col = col
                break
        
        if not order_id_col or not sku_col or not revenue_col:
            logger.error("Required columns not found")
            df = pd.DataFrame(columns=['order_id', 'sku', 'transaction_type', 'amount', 'date', 'quantity'])
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            return buffer
        
        # Build SELECT columns
        select_cols = [
            f'"{order_id_col}" as order_id',
            f'"{sku_col}" as sku',
        ]
        
        if txn_col:
            select_cols.append(f'"{txn_col}" as transaction_type')
        else:
            select_cols.append("'Shipment' as transaction_type")
        
        select_cols.append(f'ABS({revenue_col}) as amount')
        
        if date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                select_cols.append(f'CAST("{date_col}" AS DATE) as date')
            else:
                select_cols.append(f'"{date_col}" as date')
        else:
            select_cols.append("NULL as date")
        
        if quantity_col:
            select_cols.append(f'COALESCE({quantity_col}, 0) as quantity')
        else:
            select_cols.append('0 as quantity')
        
        select_clause = ', '.join(select_cols)
        
        # Build WHERE clause with filters (same logic as get_transactions)
        where_conditions = []
        
        if date_from and date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                where_conditions.append(f'CAST("{date_col}" AS DATE) >= \'{date_from}\'')
            else:
                where_conditions.append(f'"{date_col}" >= \'{date_from}\'')
        
        if date_to and date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                where_conditions.append(f'CAST("{date_col}" AS DATE) <= \'{date_to}\'')
            else:
                where_conditions.append(f'"{date_col}" <= \'{date_to}\'')
        
        if sku:
            where_conditions.append(f'"{sku_col}" = \'{sku}\'')
        
        if transaction_type and txn_col:
            where_conditions.append(f'"{txn_col}" = \'{transaction_type}\'')
        
        where_clause = ' AND '.join(where_conditions) if where_conditions else '1=1'
        
        # Query all data (no pagination for export)
        sql = f"""
        SELECT {select_clause}
        FROM sales
        WHERE {where_clause}
        ORDER BY {date_col if date_col else order_id_col} DESC
        """
        
        df = execute_query(sql)
        
        # Convert to CSV bytes
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Error exporting transactions CSV: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return empty CSV on error
        df = pd.DataFrame(columns=['order_id', 'sku', 'transaction_type', 'amount', 'date', 'quantity'])
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
