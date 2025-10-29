"""
Metrics Service - Business KPI calculations

Extracted and refactored from legacy/app.py
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from core.database import execute_query, table_exists

logger = logging.getLogger(__name__)


def calculate_transaction_revenue(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate revenue based on transaction types - uses derived fields if available
    
    Extracted from legacy/app.py:calculate_transaction_revenue()
    
    Args:
        df: DataFrame with sales data
    
    Returns:
        Dictionary with revenue metrics
    """
    if df.empty:
        return {
            'gross_revenue': 0.0,
            'refunds': 0.0,
            'free_replacement_cost': 0.0,
            'shipping_cost_loss': 0.0,
            'net_revenue': 0.0,
            'units_sold': 0,
            'orders': 0,
            'aov': 0.0,
            'transaction_breakdown': {}
        }
    
    # Check if we have derived fields from cleaning
    if 'revenue_calc' in df.columns and 'shipping_loss_calc' in df.columns:
        # Calculate directly from derived fields
        gross_revenue = df[df['revenue_calc'] > 0]['revenue_calc'].sum()
        refunds = abs(df[df['revenue_calc'] < 0]['revenue_calc'].sum())
        shipping_loss = df['shipping_loss_calc'].sum()
        units_sold = df['units_sold_calc'].sum() if 'units_sold_calc' in df.columns else 0
        
        # Get order count (handle various column names)
        order_id_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['invoice number', 'order id', 'orderid'])]
        orders = df[order_id_cols[0]].nunique() if order_id_cols else 0
        
        # Calculate FreeReplacement cost estimation (simplified)
        free_replacement_cost = 0.0
        if 'transaction_type' in df.columns:
            freereplacement_data = df[df['transaction_type'] == 'FreeReplacement']
            if not freereplacement_data.empty:
                # Estimate as average shipment revenue for FreeReplacement items
                shipment_data = df[df['transaction_type'] == 'Shipment']
                if not shipment_data.empty:
                    free_replacement_cost = freereplacement_data['revenue_calc'].sum()
        
        net_revenue = gross_revenue - refunds - shipping_loss - free_replacement_cost
        aov = net_revenue / orders if orders > 0 else 0.0
        
        # Build transaction breakdown
        breakdown = {}
        if 'transaction_type' in df.columns:
            for txn in df['transaction_type'].unique():
                if pd.notna(txn):
                    txn_data = df[df['transaction_type'] == txn]
                    breakdown[str(txn)] = {
                        'count': len(txn_data),
                        'revenue': float(txn_data['revenue_calc'].sum()),
                        'units': int(txn_data['units_sold_calc'].sum()) if 'units_sold_calc' in txn_data.columns else 0,
                    }
        
        return {
            'gross_revenue': float(gross_revenue),
            'refunds': float(refunds),
            'free_replacement_cost': float(free_replacement_cost),
            'shipping_cost_loss': float(shipping_loss),
            'net_revenue': float(net_revenue),
            'units_sold': int(units_sold),
            'orders': int(orders),
            'aov': float(aov),
            'transaction_breakdown': breakdown
        }
    
    # Fallback: use legacy calculation if derived fields not available
    if 'revenue_in_inr' in df.columns:
        gross_revenue = df['revenue_in_inr'].sum()
        units_sold = df['quantity'].sum() if 'quantity' in df.columns else 0
        order_id_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['invoice number', 'order id'])]
        orders = df[order_id_cols[0]].nunique() if order_id_cols else 0
        aov = gross_revenue / orders if orders > 0 else 0.0
        
        return {
            'gross_revenue': float(gross_revenue),
            'refunds': 0.0,
            'free_replacement_cost': 0.0,
            'shipping_cost_loss': 0.0,
            'net_revenue': float(gross_revenue),
            'units_sold': int(units_sold),
            'orders': int(orders),
            'aov': float(aov),
            'transaction_breakdown': {
                'Shipment': {
                    'count': len(df),
                    'revenue': float(gross_revenue),
                    'units': int(units_sold),
                }
            }
        }
    
    # Return zeros if no revenue columns found
    return {
        'gross_revenue': 0.0,
        'refunds': 0.0,
        'free_replacement_cost': 0.0,
        'shipping_cost_loss': 0.0,
        'net_revenue': 0.0,
        'units_sold': 0,
        'orders': 0,
        'aov': 0.0,
        'transaction_breakdown': {}
    }


def get_filtered_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    transaction_type: Optional[str] = None,
    source_file: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Get filtered data for date range and optional filters
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        transaction_type: Optional transaction type filter
        source_file: Optional source file filter
    
    Returns:
        DataFrame with filtered data or None if no data
    """
    if not table_exists('sales'):
        return None
    
    try:
        where_conditions = []
        
        # Date filtering (handle different date column names)
        date_col = None
        # Try to find date column
        column_info = execute_query("DESCRIBE sales")
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        if date_col and start_date and end_date:
            where_conditions.append(f'"{date_col}" >= \'{start_date}\'')
            where_conditions.append(f'"{date_col}" <= \'{end_date}\'')
        
        # Transaction type filter
        if transaction_type and transaction_type != "All":
            txn_col = None
            txn_columns = ['Transaction Type', 'transaction_type']
            for col in txn_columns:
                if col in column_info['column_name'].values:
                    txn_col = col
                    break
            
            if txn_col:
                where_conditions.append(f'"{txn_col}" = \'{transaction_type}\'')
        
        # Source file filter
        if source_file:
            where_conditions.append(f'source_file = \'{source_file}\'')
        
        # Build query
        where_clause = ' AND '.join(where_conditions) if where_conditions else '1=1'
        sql = f'SELECT * FROM sales WHERE {where_clause} ORDER BY "{date_col}"' if date_col else f'SELECT * FROM sales WHERE {where_clause}'
        
        df = execute_query(sql)
        return df if not df.empty else None
        
    except Exception as e:
        logger.error(f"Error getting filtered data: {e}")
        return None


def calculate_metrics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    transaction_type: Optional[str] = None,
    source_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate core business metrics (KPIs)
    
    Args:
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        transaction_type: Optional transaction type filter
        source_file: Optional source file filter
    
    Returns:
        Dictionary with KPI values
    """
    # Get filtered data
    df = get_filtered_data(start_date, end_date, transaction_type, source_file)
    
    if df is None or df.empty:
        return {
            'revenue': 0.0,
            'orders': 0,
            'avg_order_value': 0.0,
            'refunds': 0.0,
            'net_revenue': 0.0,
            'success_rate': 0.0,
            'units_sold': 0,
        }
    
    # Calculate transaction revenue
    transaction_revenue = calculate_transaction_revenue(df)
    
    return {
        'revenue': transaction_revenue['gross_revenue'],
        'orders': transaction_revenue['orders'],
        'avg_order_value': transaction_revenue['aov'],
        'refunds': transaction_revenue['refunds'],
        'net_revenue': transaction_revenue['net_revenue'],
        'success_rate': 95.0,  # Placeholder - calculate based on refund rate
        'units_sold': transaction_revenue['units_sold'],
        'gross_revenue': transaction_revenue['gross_revenue'],
        'shipping_cost_loss': transaction_revenue['shipping_cost_loss'],
        'free_replacement_cost': transaction_revenue['free_replacement_cost'],
    }


def get_revenue_trend(
    start_date: str,
    end_date: str,
    group_by: str = 'day'
) -> pd.DataFrame:
    """
    Get revenue trend data grouped by time period
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        group_by: Grouping period ('day', 'week', 'month')
    
    Returns:
        DataFrame with columns: date, revenue
    """
    # Determine date truncation based on grouping
    date_trunc_map = {
        'day': 'day',
        'week': 'week',
        'month': 'month',
    }
    
    trunc_period = date_trunc_map.get(group_by, 'day')
    
    # Try to find date and revenue columns
    column_info = execute_query("DESCRIBE sales")
    date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
    revenue_columns = ['revenue_calc', 'revenue_in_inr', 'Revenue Amount']
    txn_columns = ['Transaction Type', 'transaction_type']
    
    date_col = None
    for col in date_columns:
        if col in column_info['column_name'].values:
            date_col = col
            break
    
    revenue_col = None
    for col in revenue_columns:
        if col in column_info['column_name'].values:
            revenue_col = col
            break
    
    txn_col = None
    for col in txn_columns:
        if col in column_info['column_name'].values:
            txn_col = col
            break
    
    if not date_col or not revenue_col:
        return pd.DataFrame()
    
    # Build query with transaction type filter for shipments
    if txn_col:
        sql = f"""
        SELECT 
            DATE_TRUNC('{trunc_period}', "{date_col}") as date,
            SUM({revenue_col}) as revenue
        FROM sales
        WHERE "{txn_col}" = 'Shipment'
          AND "{date_col}" >= '{start_date}'
          AND "{date_col}" <= '{end_date}'
        GROUP BY DATE_TRUNC('{trunc_period}', "{date_col}")
        ORDER BY date
        """
    else:
        # No transaction type - use positive amounts
        sql = f"""
        SELECT 
            DATE_TRUNC('{trunc_period}', "{date_col}") as date,
            SUM({revenue_col}) as revenue
        FROM sales
        WHERE {revenue_col} > 0
          AND "{date_col}" >= '{start_date}'
          AND "{date_col}" <= '{end_date}'
        GROUP BY DATE_TRUNC('{trunc_period}', "{date_col}")
        ORDER BY date
        """
    
    return execute_query(sql)


def get_top_products(
    limit: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get top products by revenue
    
    Args:
        limit: Number of products to return
        start_date: Optional start date filter
        end_date: Optional end date filter
    
    Returns:
        DataFrame with columns: sku, revenue, orders
    """
    column_info = execute_query("DESCRIBE sales")
    sku_columns = ['Sku', 'sku', 'SKU']
    revenue_columns = ['revenue_calc', 'revenue_in_inr', 'Revenue Amount']
    txn_columns = ['Transaction Type', 'transaction_type']
    order_id_columns = ['Invoice Number', 'order_id', 'Order ID']
    date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
    
    sku_col = None
    for col in sku_columns:
        if col in column_info['column_name'].values:
            sku_col = col
            break
    
    revenue_col = None
    for col in revenue_columns:
        if col in column_info['column_name'].values:
            revenue_col = col
            break
    
    txn_col = None
    for col in txn_columns:
        if col in column_info['column_name'].values:
            txn_col = col
            break
    
    order_id_col = None
    for col in order_id_columns:
        if col in column_info['column_name'].values:
            order_id_col = col
            break
    
    date_col = None
    for col in date_columns:
        if col in column_info['column_name'].values:
            date_col = col
            break
    
    if not sku_col or not revenue_col:
        return pd.DataFrame()
    
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
        "{sku_col}" as sku,
        SUM({revenue_col}) as revenue,
        COUNT(DISTINCT "{order_id_col}") as orders
    FROM sales
    WHERE 1=1 {txn_filter} {date_filter}
    GROUP BY "{sku_col}"
    ORDER BY revenue DESC
    LIMIT {limit}
    """
    
    return execute_query(sql)
