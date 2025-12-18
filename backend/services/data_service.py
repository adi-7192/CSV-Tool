"""
Data Service - Handle raw transaction data queries with tenant isolation
"""
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from io import BytesIO
from core.database import execute_query, table_exists
import logging
import re

from utils.error_handler import handle_service_error, log_error
from utils.tenant_filter import get_tenant_filter_sql

logger = logging.getLogger(__name__)


def detect_order_id_column(column_info: pd.DataFrame) -> Optional[str]:
    """
    Detect the column that contains order IDs.
    
    Prioritizes "Order Id" column and explicitly excludes "Invoice Number".
    
    Args:
        column_info: DataFrame from DESCRIBE sales query
        
    Returns:
        Column name that contains order IDs, or None if not found
    """
    available_cols = list(column_info['column_name'].values)
    
    logger.info(f"Available columns for order_id detection: {available_cols}")
    
    # EXPLICITLY check for "Order Id" first (with lowercase 'd')
    # This is the column that contains real order IDs like "PM02-190", "MAA4-154"
    if 'Order Id' in available_cols:
        logger.info("Found 'Order Id' column - using it for order_id")
        return 'Order Id'
    
    # Then check for "Order ID" (with uppercase 'D')
    if 'Order ID' in available_cols:
        logger.info("Found 'Order ID' column - using it for order_id")
        return 'Order ID'
    
    # Then check for lowercase "order_id"
    if 'order_id' in available_cols:
        logger.info("Found 'order_id' column - using it for order_id")
        return 'order_id'
    
    # Check for other order-related columns (but NOT "Invoice Number")
    order_id_candidates = []
    priority_columns = ['order_number', 'Order Number']
    for col in priority_columns:
        if col in available_cols:
            order_id_candidates.append(col)
    
    # Also check for any column with "order" in the name (case-insensitive)
    # EXPLICITLY EXCLUDE "Invoice Number" and any column with "invoice" in the name
    for col in available_cols:
        col_lower = col.lower()
        if 'order' in col_lower and col not in order_id_candidates:
            # Skip columns that are clearly not order IDs
            # EXPLICITLY EXCLUDE "Invoice Number" and any invoice-related columns
            if 'date' not in col_lower and 'amount' not in col_lower and 'invoice' not in col_lower:
                order_id_candidates.append(col)
    
    # If we have candidates, try to find the one with real order IDs
    if order_id_candidates:
        # Try each candidate by sampling data
        candidate_cols = [f'"{c}"' for c in order_id_candidates[:5]]
        sample_sql = f'SELECT {", ".join(candidate_cols)} FROM sales LIMIT 10'
        try:
            sample_df = execute_query(sample_sql)
            if not sample_df.empty:
                # Order IDs can be in various formats:
                # - Amazon format: "171-1269513-1525149" (numbers-dashes-numbers)
                # - Other formats: "PM02-190", "MAA4-154", "LK01-129" (letters-numbers-dashes-numbers)
                # Pattern: alphanumeric-dashes-alphanumeric (flexible)
                order_id_pattern = re.compile(r'^[A-Za-z0-9]+-[A-Za-z0-9]+(-[A-Za-z0-9]+)*$')
                
                for candidate in order_id_candidates:
                    if candidate in sample_df.columns:
                        # Check if this column has order ID format
                        sample_values = sample_df[candidate].dropna().astype(str).head(10)
                        
                        # Count valid order IDs (format: alphanumeric-dashes-alphanumeric)
                        valid_id_count = sum(1 for val in sample_values if order_id_pattern.match(str(val).strip()))
                        
                        # Count synthetic IDs (start with "UNKNOWN")
                        synthetic_id_count = sum(1 for val in sample_values if str(val).strip().startswith('UNKNOWN'))
                        
                        # Prefer columns with valid order IDs and avoid columns with synthetic IDs
                        if valid_id_count >= 2 or (valid_id_count > 0 and synthetic_id_count == 0):
                            logger.info(f"Selected order_id column '{candidate}' (contains valid order IDs: {valid_id_count}/{len(sample_values)})")
                            return candidate
                
                # If no column matched order ID pattern, use first priority column
                logger.info(f"Using first available order_id column '{order_id_candidates[0]}'")
                return order_id_candidates[0]
        except Exception as e:
            logger.warning(f"Could not sample data to detect order_id column: {e}")
            # Fallback to first candidate
            return order_id_candidates[0] if order_id_candidates else None
    
    # Final fallback: check for any order-related column (but NOT "Invoice Number")
    for col in available_cols:
        col_lower = col.lower()
        if 'order' in col_lower and 'invoice' not in col_lower and 'date' not in col_lower and 'amount' not in col_lower:
            logger.info(f"Using fallback order_id column '{col}'")
            return col
    
    logger.error("Could not find any order_id column (excluding Invoice Number)")
    return None


def get_transactions(
    page: int = 1,
    limit: int = 50,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sku: Optional[str] = None,
    transaction_type: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get paginated transaction data from sales table with tenant isolation.
    
    Args:
        page: Page number (1-indexed)
        limit: Number of rows per page
        date_from: Start date filter
        date_to: End date filter
        sku: SKU filter
        transaction_type: Transaction type filter
        tenant_id: User's tenant ID for data isolation
    
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
        
        # Use helper function to detect order_id column
        order_id_col = detect_order_id_column(column_info)
        
        # DEBUG: Log which column was selected
        logger.info(f"🔍 DEBUG: Selected order_id column: '{order_id_col}'")
        print(f"\n🔍 DEBUG: Selected order_id column: '{order_id_col}'\n")
        
        # DEBUG: Log all available columns
        available_cols = list(column_info['column_name'].values)
        logger.info(f"🔍 DEBUG: All available columns: {available_cols}")
        print(f"🔍 DEBUG: All available columns: {available_cols}\n")
        
        # DEBUG: Check if "Order Id" and "Invoice Number" exist
        if 'Order Id' in available_cols:
            logger.info("✅ DEBUG: 'Order Id' column EXISTS in database")
            print("✅ DEBUG: 'Order Id' column EXISTS in database\n")
        else:
            logger.warning("⚠️ DEBUG: 'Order Id' column NOT FOUND in database")
            print("⚠️ DEBUG: 'Order Id' column NOT FOUND in database\n")
        
        if 'Invoice Number' in available_cols:
            logger.warning("⚠️ DEBUG: 'Invoice Number' column EXISTS in database")
            print("⚠️ DEBUG: 'Invoice Number' column EXISTS in database\n")
        else:
            logger.info("✅ DEBUG: 'Invoice Number' column NOT FOUND in database")
            print("✅ DEBUG: 'Invoice Number' column NOT FOUND in database\n")
        
        # DEBUG: Sample data from both columns if they exist
        if 'Order Id' in available_cols and 'Invoice Number' in available_cols:
            try:
                sample_sql = 'SELECT "Order Id", "Invoice Number" FROM sales LIMIT 5'
                sample_df = execute_query(sample_sql)
                if not sample_df.empty:
                    logger.info(f"🔍 DEBUG: Sample data from 'Order Id': {sample_df['Order Id'].head(3).tolist()}")
                    logger.info(f"🔍 DEBUG: Sample data from 'Invoice Number': {sample_df['Invoice Number'].head(3).tolist()}")
                    print(f"🔍 DEBUG: Sample data from 'Order Id': {sample_df['Order Id'].head(3).tolist()}\n")
                    print(f"🔍 DEBUG: Sample data from 'Invoice Number': {sample_df['Invoice Number'].head(3).tolist()}\n")
            except Exception as e:
                logger.warning(f"Could not sample data for comparison: {e}")
        
        sku_columns = ['sku', 'Sku', 'SKU']
        txn_columns = ['transaction_type', 'Transaction Type']
        revenue_columns = ['revenue_amount', 'Invoice Amount', 'revenue_in_inr']
        date_columns = ['order_date', 'Invoice Date', 'invoice_date', 'Order Date']
        quantity_columns = ['quantity', 'Quantity', 'units_sold']
        
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
        
        # TENANT ISOLATION: Always filter by tenant_id first
        if tenant_id:
            where_conditions.append(get_tenant_filter_sql(tenant_id))
        
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
        
        # DEBUG: Log the SQL query
        logger.info(f"🔍 DEBUG: Executing SQL query:\n{sql}")
        print(f"\n🔍 DEBUG: Executing SQL query:\n{sql}\n")
        
        df = execute_query(sql)
        
        # DEBUG: Log sample results
        if not df.empty:
            logger.info(f"🔍 DEBUG: Query returned {len(df)} rows")
            logger.info(f"🔍 DEBUG: Sample order_id values: {df['order_id'].head(5).tolist()}")
            print(f"🔍 DEBUG: Query returned {len(df)} rows")
            print(f"🔍 DEBUG: Sample order_id values: {df['order_id'].head(5).tolist()}\n")
        
        # Convert DataFrame to list of dicts
        data = []
        for _, row in df.iterrows():
            order_id_value = str(row.get('order_id', ''))
            # DEBUG: Log first few order_id values being returned
            if len(data) < 3:
                logger.info(f"🔍 DEBUG: Returning order_id value: '{order_id_value}'")
                print(f"🔍 DEBUG: Returning order_id value: '{order_id_value}'\n")
            
            data.append({
                "order_id": order_id_value,
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
        
    except ValueError as e:
        log_error(e, 'get_transactions', {'page': page, 'limit': limit, 'date_from': date_from, 'date_to': date_to, 'sku': sku, 'transaction_type': transaction_type})
        return {
            "data": [],
            "total": 0,
            "page": page,
            "total_pages": 0,
            "error": True,
            "message": f"Invalid input: {str(e)}",
            "status": 400
        }
    except Exception as e:
        log_error(e, 'get_transactions', {'page': page, 'limit': limit, 'date_from': date_from, 'date_to': date_to, 'sku': sku, 'transaction_type': transaction_type})
        return {
            "data": [],
            "total": 0,
            "page": page,
            "total_pages": 0,
            "error": True,
            "message": "Failed to fetch transactions. Please try again later.",
            "status": 500
        }


def get_unique_skus(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get list of unique SKU values from sales table with tenant isolation.
    
    Args:
        tenant_id: User's tenant ID for data isolation
    
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
        
        # Build tenant filter
        tenant_filter = get_tenant_filter_sql(tenant_id) if tenant_id else '1=1'
        
        # Query distinct SKUs with tenant isolation
        sql = f"""
        SELECT DISTINCT "{sku_col}" as sku
        FROM sales
        WHERE {tenant_filter} AND "{sku_col}" IS NOT NULL AND TRIM("{sku_col}") != ''
        ORDER BY "{sku_col}" ASC
        """
        
        df = execute_query(sql)
        
        # Convert to list
        skus = [str(row['sku']) for _, row in df.iterrows() if row['sku']]
        
        return {
            "skus": skus,
        }
        
    except ValueError as e:
        log_error(e, 'get_unique_skus', {})
        return {
            "skus": [],
            "error": True,
            "message": f"Invalid input: {str(e)}",
            "status": 400
        }
    except Exception as e:
        log_error(e, 'get_unique_skus', {})
        return {
            "skus": [],
            "error": True,
            "message": "Failed to fetch unique SKUs. Please try again later.",
            "status": 500
        }


def get_data_statistics(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get data statistics: total records, date range, unique SKUs with tenant isolation.
    
    Args:
        tenant_id: User's tenant ID for data isolation
    
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
        
        # Build tenant filter
        tenant_filter = get_tenant_filter_sql(tenant_id) if tenant_id else '1=1'
        
        # Get total records (with tenant isolation)
        count_sql = f"SELECT COUNT(*) as total FROM sales WHERE {tenant_filter}"
        count_df = execute_query(count_sql)
        total_records = int(count_df['total'].iloc[0]) if not count_df.empty else 0
        
        # Get date range (with tenant isolation)
        date_start = None
        date_end = None
        if date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                range_sql = f"SELECT MIN(CAST(\"{date_col}\" AS DATE)) as min_date, MAX(CAST(\"{date_col}\" AS DATE)) as max_date FROM sales WHERE {tenant_filter} AND \"{date_col}\" IS NOT NULL"
            else:
                range_sql = f"SELECT MIN(\"{date_col}\") as min_date, MAX(\"{date_col}\") as max_date FROM sales WHERE {tenant_filter} AND \"{date_col}\" IS NOT NULL"
            
            range_df = execute_query(range_sql)
            if not range_df.empty:
                min_date = range_df['min_date'].iloc[0]
                max_date = range_df['max_date'].iloc[0]
                if min_date is not None:
                    date_start = str(min_date)
                if max_date is not None:
                    date_end = str(max_date)
        
        # Get unique SKUs count (with tenant isolation)
        unique_skus = 0
        if sku_col:
            sku_sql = f"SELECT COUNT(DISTINCT \"{sku_col}\") as count FROM sales WHERE {tenant_filter} AND \"{sku_col}\" IS NOT NULL AND TRIM(\"{sku_col}\") != ''"
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
        
    except ValueError as e:
        log_error(e, 'get_data_statistics', {})
        return {
            "total_records": 0,
            "date_range": {"start": None, "end": None},
            "unique_skus": 0,
            "error": True,
            "message": f"Invalid input: {str(e)}",
            "status": 400
        }
    except Exception as e:
        log_error(e, 'get_data_statistics', {})
        return {
            "total_records": 0,
            "date_range": {"start": None, "end": None},
            "unique_skus": 0,
            "error": True,
            "message": "Failed to fetch data statistics. Please try again later.",
            "status": 500
        }


def export_transactions_csv(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sku: Optional[str] = None,
    transaction_type: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> BytesIO:
    """
    Export filtered transactions to CSV with tenant isolation.
    
    Args:
        date_from: Start date filter
        date_to: End date filter
        sku: SKU filter
        transaction_type: Transaction type filter
        tenant_id: User's tenant ID for data isolation
    
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
        
        # Use helper function to detect order_id column
        order_id_col = detect_order_id_column(column_info)
        
        sku_col = None
        txn_col = None
        revenue_col = None
        date_col = None
        quantity_col = None
        
        sku_columns = ['sku', 'Sku', 'SKU']
        txn_columns = ['transaction_type', 'Transaction Type']
        revenue_columns = ['revenue_amount', 'Invoice Amount', 'revenue_in_inr']
        date_columns = ['order_date', 'Invoice Date', 'invoice_date', 'Order Date']
        quantity_columns = ['quantity', 'Quantity', 'units_sold']
        
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
        
        # TENANT ISOLATION: Always filter by tenant_id first
        if tenant_id:
            where_conditions.append(get_tenant_filter_sql(tenant_id))
        
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
